"""dctjoin.decay — replicate a *decaying* note (piano, harp, mallets): flatten, loop, re-envelope.

A decaying note has no steady body to loop.  sfzc's answer, used here with the untouched-loop
machinery: divide the body by its own smooth envelope so it becomes a stationary sound at the
level it had at the join, loop *that* with dctloop, bridge the recorded attack onto the loop,
and hand the decay back to the SFZ amplitude envelope.  sfizz's ``ampeg_decay`` is a pure
exponential exp(-9 t / T) (sustain 0), so the original envelope, relative to the file's, is
fitted as a sum of up to three exponentials and each term becomes one region of the same
sample: all regions play sample-synchronously, so their gains add.  ``ampeg_hold`` covers the
recorded attack, so nothing is re-shaped until the loop takes over.

    file  = [recorded attack ........][bridge][untouched loop of the flattened body]
    gain  = 1 ------------------------- hold ->  sum_i a_i exp(-(t - t_h) / tau_i),  sum a_i = 1

The loop level is the recording's level at the join, so the bridge has (almost) nothing to
ramp and the envelope is exactly 1 there.  Because the SSO piano samples are cut before the
note has died, the loop + envelope continues the decay past the recording's end.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf
from scipy.optimize import nnls

from dctloop import f0_per_channel, load_audio, loop_signal, measure, note_from_name

from .bridge import bridge_attack, continuity_metrics
from .join import raised_cosine, splice
from .segment import segment_note
from .sfz import key_and_tune
from .unaltered import _rms, encode_ogg, pad_after_loop, release_time, sfizz_gain_db

SFZ_EXP = 9.0                      # sfizz: ampeg_decay T  <->  exp(-9 t / T)


# ------------------------------------------------------------------------------- envelope

def smooth_envelope(x: np.ndarray, fs: int, win_s: float = 0.03, hop_s: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """(sample positions, RMS of the channel mean) on a hop grid, Hann-smoothed over ~3 windows."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    w, h = max(8, int(win_s * fs)), max(1, int(hop_s * fs))
    frames = np.lib.stride_tricks.sliding_window_view(mono, w)[::h]
    e = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)
    k = np.hanning(7)
    k /= k.sum()
    e = np.convolve(np.pad(e, 3, mode='edge'), k, mode='valid')
    return np.arange(len(frames)) * h + w // 2, e


def flatten(x: np.ndarray, fs: int, start: int, end: int, ref: int) -> tuple[np.ndarray, np.ndarray]:
    """x[start:end] divided by its smooth envelope and scaled to the envelope at sample ``ref``:
    a stationary version of the body at the join level.  Returns (flat, gain per sample)."""
    pos, e = smooth_envelope(x, fs)
    env = np.interp(np.arange(start, end), pos, e)
    e_ref = float(np.interp(ref, pos, e))
    g = e_ref / np.maximum(env, 1e-9)
    return x[start:end] * g[:, None], g


def fit_decay(t: np.ndarray, g: np.ndarray, n_max: int = 3, tau_grid: np.ndarray | None = None,
              w: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Fit g(t) (t >= 0, g(0) = 1) as sum_i a_i exp(-t / tau_i), a_i >= 0, sum a_i = 1, with at most
    ``n_max`` terms: non-negative least squares over a log grid of tau, then the strongest terms
    refitted.  Returns [(a_i, tau_i)]."""
    if tau_grid is None:
        tau_grid = np.geomspace(0.03, 60.0, 60)
    if w is None:
        w = np.ones_like(t)
    lg = np.log(np.maximum(g, 1e-6))                     # fit in dB-ish space, weighted back to amplitude
    def solve(taus):
        A = np.exp(-t[:, None] / taus[None, :])
        # constraint sum a = 1 as a heavily weighted row; weights favour the loud, early part
        Aw = np.concatenate([A * np.sqrt(w)[:, None], np.full((1, len(taus)), 30.0)], axis=0)
        yw = np.concatenate([g * np.sqrt(w), [30.0]])
        a, _ = nnls(Aw, yw)
        return a
    a = solve(tau_grid)
    # distinct time constants: local maxima of the coefficient vector over the tau grid, at
    # least a factor of two apart, strongest first (NNLS on a fine grid otherwise spends its
    # terms on two adjacent grid points of the same decay)
    pk = [i for i in range(len(a)) if a[i] > 0 and a[i] >= a[max(0, i - 1)] and a[i] >= a[min(len(a) - 1, i + 1)]]
    pk.sort(key=lambda i: -a[i])
    keep = []
    for i in pk:
        if all(abs(math.log(tau_grid[i] / tau_grid[j])) > math.log(2.0) for j in keep):
            keep.append(i)
        if len(keep) == n_max:
            break
    if not keep:
        keep = [int(np.argmax(a))]
    taus = tau_grid[keep]
    a = solve(taus)
    s = a.sum()
    if s > 0:
        a = a / s
    return sorted(((float(ai), float(ti)) for ai, ti in zip(a, taus) if ai > 1e-4), key=lambda p: p[1])


def envelope_error_db(t: np.ndarray, g: np.ndarray, comps: list[tuple[float, float]]) -> tuple[float, float]:
    """(rms, max) error in dB of the fitted envelope against g over t."""
    fit = sum(a * np.exp(-t / tau) for a, tau in comps)
    d = 20 * np.log10(np.maximum(fit, 1e-9) / np.maximum(g, 1e-9))
    return float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d)))


# ------------------------------------------------------------------------------- the whole thing

@dataclass
class DecayReplication:
    source: str
    fs: int
    basis: str
    f0: list
    f0_used: float
    pitch_method: str
    keycenter: int
    tune_cents: float
    join_s: float
    attack_s: float
    hold_s: float                # ampeg_hold: the envelope is 1 until here (the recorded attack + bridge start)
    body_s: list                 # [start, end] of the flattened body dctloop analysed, seconds into the file
    env_end_s: float             # the recording is a note until here; the envelope is fitted up to it
    body_fall_db: float          # how far the original had decayed over the analysed body
    loop_seconds: float
    L: int
    K: int
    loop_start: int
    loop_end: int
    loop_untouched: bool
    envelope: list               # [(a_i, tau_i)]  -> regions with volume 20 log a_i, ampeg_decay 9 tau_i
    envelope_fit_db: list        # [rms, max] error of the fit over the recording
    release_sfz: float
    bridge: dict | None
    continuity: dict
    loop_metrics: dict
    render_env_err_db: list | None   # sfizz render vs original: [rms, max] envelope error over the recording's length
    outputs: dict


def replicate_decaying(path: str, out_dir: str | None = None, seconds: float = 0.5, *, basis: str = 'dft',
                       f0: float | None = None, use_hint: bool = True, lock: float = 1.5, max_attack_s: float = 0.5,
                       bridge_s: float = 0.15, body_floor_db: float = -20.0, n_exp: int = 3, xfade_s: float = 0.01,
                       format: str = 'flac', bits: int = 16, quality: float = 1.0, release: float | None = None,
                       preview: bool = True, sfizz: bool = True, stem: str | None = None, hint_mult: float = 1.0,
                       verbose: bool = False) -> DecayReplication:
    x, fs = load_audio(path)
    name = os.path.splitext(os.path.basename(path))[0]
    stem = stem or name
    seg = segment_note(x, fs)
    pos, env = smooth_envelope(x, fs)
    env_db = 20 * np.log10(env + 1e-12)
    peak_db = float(env_db.max())

    # ---- join: the end of the attack budget (the fast part of a piano's decay stays recorded)
    J = seg.onset + int(round(max_attack_s * fs))
    Bn = int(round(bridge_s * fs))
    J = max(J, seg.attack_end + Bn)
    e_J = float(np.interp(J, pos, env))
    e_J_db = 20 * math.log10(e_J + 1e-12)
    # ---- the recording is usable until it has fallen ``env_floor_db`` below its level at the
    #      join or reaches its (faded) end: that is the range the envelope is fitted over
    # the recording's real content end: last frame within 70 dB of the peak, minus the fade
    alive = np.where(env_db > peak_db - 70.0)[0]
    last = int(min(pos[alive[-1]] - int(0.05 * fs), len(x) - 1)) if alive.size else seg.end
    below = np.where((pos > J) & (env_db < e_J_db - 40.0))[0]
    env_end = int(min(pos[below[0]] if below.size else last, last))
    env_end = max(env_end, J + int(0.5 * fs))
    # ---- the loop's analysis body is the first part of that, before the timbre has drifted:
    #      until the level is body_floor_db (default 20 dB) below the join, but at least 1 s
    below = np.where((pos > J) & (env_db < e_J_db + body_floor_db))[0]
    end = int(min(max(pos[below[0]] if below.size else env_end, J + int(1.0 * fs)), env_end))
    if end - J < int(0.25 * fs):
        raise ValueError(f'{name}: only {(end - J) / fs:.2f}s of body after the join')
    flat, g_flat = flatten(x, fs, J, end, J)

    # ---- pitch on the flattened body, loop it (untouched), bridge the attack onto it
    hint = f0 or note_from_name(name)
    if hint and not f0:
        hint = hint * hint_mult                      # e.g. 2.0 for a library whose names sit an octave low
    f0c, pinfo = f0_per_channel(flat, fs, f0, hint, use_hint=use_hint, detail=True)
    good = f0c[np.isfinite(f0c)]
    if not good.size:
        raise ValueError(f'{name}: no pitch found')
    f0_used = float(np.exp(np.mean(np.log(good))))
    f0_list = [float(v) if np.isfinite(v) else f0_used for v in f0c]
    loop, linfo = loop_signal(flat, fs, seconds, basis=basis, f0=f0_list, lock=lock)
    lp_peak = float(np.max(np.abs(loop)))
    if lp_peak > 0.999:
        loop = loop * (0.999 / lp_peak)
    L = len(loop)
    seg_used = flat[linfo['analysis_offset']: linfo['analysis_offset'] + linfo['N']]
    lm = measure(seg_used, fs, loop, f0_list)
    xb, binfo = bridge_attack(x, fs, J, loop, f0_list, bridge_s=Bn / fs)
    out, ls, le, X = splice(xb, seg.onset, J, loop, fs, xfade_s=xfade_s, f0=f0_used)
    untouched = bool(np.array_equal(out[ls: le + 1], loop))
    cm = continuity_metrics(out, x, J, ls, fs, f0_list, loop=loop)

    # ---- the envelope the SFZ has to restore: original / file, from the hold point on
    t_h = binfo['bridge_start'] - seg.onset                      # hold covers the untouched attack
    pos_o, env_o = smooth_envelope(x[seg.onset:], fs)
    pos_f, env_f = smooth_envelope(out, fs)
    tt = np.arange(t_h, env_end - seg.onset, int(0.005 * fs))      # the whole usable recording
    eo = np.interp(tt, pos_o, env_o)
    # the file is attack + one loop; when played it repeats the loop, so beyond the file its
    # envelope is the loop's level: np.interp holds the last value, which is that level
    ef = np.interp(tt, pos_f, env_f)
    g = eo / np.maximum(ef, 1e-9)
    g = g / g[0]
    w = 1.0 / np.maximum(g, 1e-3) ** 2                          # relative error: every dB counts the same
    comps = fit_decay((tt - t_h) / fs, g, n_max=n_exp, w=w)
    fit_err = envelope_error_db((tt - t_h) / fs, g, comps)
    T_rel = release if release is not None else min(release_time(x, fs, seg.release_onset, seg.end)[0], 0.5)
    key, tune = key_and_tune(f0_used)

    outputs, render_err = {}, None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ext = {'flac': 'flac', 'wav': 'wav', 'ogg': 'ogg'}[format]
        p_audio = os.path.join(out_dir, f'{stem}.{ext}')
        peak = float(np.max(np.abs(out)))
        out_w = out * (0.999 / peak) if peak > 0.999 else out
        if format == 'ogg':
            tmp = os.path.join(out_dir, f'{stem}__tmp.wav')
            sf.write(tmp, pad_after_loop(out_w, ls, le, fs), fs, subtype='PCM_24')
            encode_ogg(tmp, p_audio, quality)
            os.remove(tmp)
        else:
            sf.write(p_audio, out_w, fs, subtype=f'PCM_{bits}')
        outputs['audio'] = p_audio
        if sfizz:
            p_tmp = os.path.join(out_dir, f'{stem}__render.sfz')
            write_decay_sfz(p_tmp, [dict(sample=f'{stem}.{ext}', keycenter=key, tune_cents=0.0, loop_start=ls, loop_end=le,
                                          hold=t_h / fs, envelope=comps, release=T_rel)], header='temporary render',
                            veltrack0=True)
            try:
                import pysfizz
                syn = pysfizz.Synth(sample_rate=fs)
                if syn.load_sfz_file(os.path.abspath(p_tmp)):
                    note_on = max(len(x) / fs, 2.0)
                    y = np.asarray(syn.render_note(key, 100, note_on, note_on + T_rel * 1.5 + 0.2), dtype=np.float64).T
                    y *= 10 ** (-sfizz_gain_db(fs) / 20)
                    pos_r, env_r = smooth_envelope(y, fs)
                    n_cmp = min(env_end - seg.onset, len(y))            # up to where the recording is still a note
                    tt2 = np.arange(int(0.05 * fs), n_cmp, int(0.01 * fs))
                    d = 20 * np.log10(np.interp(tt2, pos_r, env_r) / np.maximum(np.interp(tt2, pos_o, env_o), 1e-9))
                    render_err = [float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d)))]
                    outputs['sfizz_render'] = os.path.join(out_dir, f'{stem}_sfizz.wav')
                    if preview:
                        o = x[seg.onset:]
                        gap = np.zeros((int(0.5 * fs), 2))
                        pv = np.concatenate([o, gap, y[: len(o) + int(1.5 * fs)]], axis=0)
                        pv *= 0.9 / max(np.max(np.abs(pv)), 1e-9)
                        p_prev = os.path.join(out_dir, f'{stem}_preview.wav')
                        sf.write(p_prev, pv, fs, subtype='PCM_16')
                        outputs['preview'] = p_prev
                    outputs.pop('sfizz_render')
            finally:
                os.remove(p_tmp)

    rep = DecayReplication(source=path, fs=fs, basis=basis, f0=f0_list, f0_used=f0_used, pitch_method=pinfo.get('method', '?'),
                           keycenter=key, tune_cents=float(tune), join_s=J / fs, attack_s=(J - seg.onset) / fs, hold_s=t_h / fs,
                           body_s=[J / fs, end / fs], env_end_s=env_end / fs, body_fall_db=float(20 * np.log10(e_J / max(np.interp(end, pos, env), 1e-12))),
                           loop_seconds=linfo['seconds'], L=L, K=linfo['K'], loop_start=ls, loop_end=le, loop_untouched=untouched,
                           envelope=[list(c) for c in comps], envelope_fit_db=list(fit_err), release_sfz=T_rel, bridge=binfo,
                           continuity=cm, loop_metrics=lm, render_env_err_db=render_err, outputs=outputs)
    if out_dir:
        with open(os.path.join(out_dir, f'{stem}.json'), 'w') as fh:
            json.dump(asdict(rep), fh, default=float)
        outputs['json'] = os.path.join(out_dir, f'{stem}.json')
    if verbose:
        print(summary_line(rep))
    return rep


def region_lines_decay(sample: str, keycenter: int, tune_cents: float, loop_start: int, loop_end: int, hold: float,
                       envelope: list, release: float, lokey: int | None = None, hikey: int | None = None,
                       lovel: int = 1, hivel: int = 127, sw_last: int | None = None) -> list[str]:
    """One <region> per envelope component, same sample, same key/velocity range: gains add."""
    lo = keycenter if lokey is None else lokey
    hi = keycenter if hikey is None else hikey
    sw = f' sw_last={sw_last}' if sw_last is not None else ''
    lines = []
    for a, tau in envelope:
        T = float(np.clip(SFZ_EXP * tau, 0.001, 100.0))
        lines.append(f'<region> sample={sample} pitch_keycenter={keycenter} lokey={lo} hikey={hi} lovel={lovel} hivel={hivel} '
                     f'tune={int(round(tune_cents))} loop_start={loop_start} loop_end={loop_end} '
                     f'volume={20 * math.log10(max(a, 1e-6)):.2f} ampeg_hold={hold:.3f} ampeg_decay={T:.3f} ampeg_sustain=0 '
                     f'ampeg_release={release:.3f}{sw}')
    return lines


def write_decay_sfz(path: str, regions: list[dict], header: str = '', veltrack0: bool = False) -> str:
    lines = [f'// {header}' if header else '// dctjoin decay replication', '<control>', 'default_path=',
             '<global>', 'loop_mode=loop_continuous' + (' amp_veltrack=0' if veltrack0 else '')]
    for r in regions:
        lines += region_lines_decay(os.path.basename(r['sample']), r['keycenter'], r['tune_cents'], r['loop_start'], r['loop_end'],
                                    r['hold'], r['envelope'], r['release'], r.get('lokey'), r.get('hikey'),
                                    r.get('lovel', 1), r.get('hivel', 127))
    text = '\n'.join(lines) + '\n'
    with open(path, 'w') as fh:
        fh.write(text)
    return text


def summary_line(r: DecayReplication) -> str:
    env = ' + '.join(f'{a:.2f}e^(-t/{tau:.2f}s)' for a, tau in r.envelope)
    rr = f'{r.render_env_err_db[0]:.1f}/{r.render_env_err_db[1]:.1f} dB' if r.render_env_err_db else 'n/a'
    return (f'  [{os.path.splitext(os.path.basename(r.source))[0]}] f0={r.f0_used:.1f}Hz/{r.pitch_method} key={r.keycenter} '
            f'attack={r.attack_s:.2f}s hold={r.hold_s:.2f}s body={r.body_s[1] - r.body_s[0]:.2f}s (fell {r.body_fall_db:.0f} dB) '
            f'loop={r.loop_seconds:.3f}s untouched={"yes" if r.loop_untouched else "NO"} tail ncc={r.continuity.get("tail_ncc", float("nan")):.3f} '
            f'| env {env} fit {r.envelope_fit_db[0]:.1f}/{r.envelope_fit_db[1]:.1f} dB | sfizz render vs original {rr}')
