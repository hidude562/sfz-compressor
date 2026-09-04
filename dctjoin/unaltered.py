"""dctjoin.unaltered — join a recorded attack to a dctloop loop without touching the loop.

The loop goes into the output file byte-for-byte as dctloop produced it (``loop_untouched`` in
the result is an explicit check).  Everything adapts on the recording side:

1. ``find_join``          The loop's phases at its sample 0 are fixed, so instead of moving them
                          to the join we move the join to them: search the recording after the
                          attack for the instant J whose surroundings best match the loop's entry
                          — normalised cross-correlation of x[J-W:J+W] with the loop's wrap-around
                          window loop[L-W:] + loop[:W] (W = a few periods), per channel, with a
                          small penalty for a level mismatch at J.
2. ``fit_attack_level``   Raised-cosine gain ramp over the last part of the attack so that its
                          RMS arrives at the loop's RMS at J, per channel, clipped to +-6 dB.  The
                          ramp never covers the first half of the attack, so the transient is left
                          alone.
3. ``morph_tail``         (dctjoin.join) EQ the attack's tail towards the loop's timbre.
4. ``splice``             (dctjoin.join) short raised-cosine cross-fade from the recording into
                          loop[L-X:], the samples that cyclically precede loop[0], then the loop.
                          The fade lives in the attack part of the file; loop_start points at
                          loop[0].

Then: one FLAC (attack + loop), an .sfz with the loop points and a release time fitted from the
recording's own tail, an sfizz render, and an A/B preview (original note, gap, replication).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf
from scipy.signal import correlate

from dctloop import f0_per_channel, find_body, load_audio, loop_signal, measure, note_from_name

from .bridge import bridge_attack, continuity_metrics, find_join_bridge
from .join import junction_metrics, morph_tail, raised_cosine, splice
from .segment import rms_envelope_db, segment_note
from .sfz import key_and_tune, write_sfz


def _rms(a: np.ndarray, axis=None):
    return np.sqrt(np.mean(np.square(a), axis=axis)) + 1e-12


# ------------------------------------------------------------------------------- 1. the join point

def find_join(x: np.ndarray, loop: np.ndarray, fs: int, f0: float, lo: int, hi: int,
              periods: float = 4.0, level_weight: float = 0.05) -> dict:
    """Best join J in [lo, hi] for an untouched loop.

    score(J) = NCC(x[J-W:J+W], loop[L-W:]+loop[:W]) - level_weight * |level step at J in dB|

    Returns dict(J, ncc, level_db, W, n_candidates) where level_db is the recording's level in the
    W samples before J relative to the loop's first W samples (positive: recording louder).
    """
    if loop.ndim == 1:
        loop = loop[:, None]
    if x.ndim == 1:
        x = x[:, None]
    L, C = loop.shape
    W = int(round(periods * fs / max(f0, 20.0)))
    W = int(min(max(W, int(0.005 * fs)), L // 2))
    lo = int(max(lo, W))
    hi = int(min(hi, len(x) - W - 1))
    if hi <= lo:
        raise ValueError(f'no room to search for a join (lo={lo}, hi={hi}, W={W})')
    n = hi - lo + 1
    tmpl = np.concatenate([loop[L - W:], loop[:W]], axis=0)            # what surrounds loop[0]
    seg = x[lo - W: hi + W]                                             # window for J=lo starts at 0
    ncc = np.zeros(n)
    e_before = np.zeros(n)
    for c in range(C):
        s, t = seg[:, c], tmpl[:, c]
        dots = correlate(s, t, mode='valid', method='fft')[:n]          # sum s[k+m] t[m], k = J - lo
        cs = np.concatenate([[0.0], np.cumsum(s * s)])
        en = cs[2 * W: 2 * W + n] - cs[:n]                              # energy of x[J-W:J+W]
        ncc += dots / (np.sqrt(np.maximum(en, 1e-30)) * (np.linalg.norm(t) + 1e-12))
        e_before += cs[W: W + n] - cs[:n]                               # energy of x[J-W:J]
    ncc /= C
    e_loop = float(np.sum(loop[:W] ** 2))
    level_db = 10 * np.log10((e_before + 1e-30) / (e_loop + 1e-30))
    score = ncc - level_weight * np.abs(level_db)
    k = int(np.argmax(score))
    return dict(J=lo + k, ncc=float(ncc[k]), level_db=float(level_db[k]), W=W, n_candidates=n,
                score=float(score[k]), best_ncc=float(ncc.max()))


def _ncc_at(x: np.ndarray, loop: np.ndarray, J: int, W: int) -> float:
    """Waveform correlation of the recording's last W samples before J with the loop's last W
    samples (what cyclically precedes loop[0]), mean over channels: does the tail arrive on the
    loop's entry?"""
    L = len(loop)
    t = loop[L - W:]
    s = x[J - W: J]
    return float(np.mean([np.dot(s[:, c], t[:, c]) / (np.linalg.norm(s[:, c]) * np.linalg.norm(t[:, c]) + 1e-12)
                          for c in range(x.shape[1])]))


# ------------------------------------------------------------------------------- 2. level

def fit_attack_level(x: np.ndarray, onset: int, J: int, loop: np.ndarray, fs: int, f0: float,
                     ramp_s: float = 0.08, max_db: float = 6.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Copy of ``x`` whose [onset:J] tail is gain-ramped so its per-channel RMS at J equals the
    loop's.  Returns (x2, gains, ramp_samples).  The loop is not touched."""
    if loop.ndim == 1:
        loop = loop[:, None]
    W = int(np.clip(4 * fs / max(f0, 20.0), 0.02 * fs, 0.05 * fs))
    W = int(min(W, J - onset, len(loop)))
    g = _rms(loop[:W], axis=0) / _rms(x[J - W: J], axis=0)
    g = np.clip(g, 10 ** (-max_db / 20), 10 ** (max_db / 20))
    R = int(min(ramp_s * fs, 0.5 * (J - onset)))
    x2 = x.copy()
    if R > 1:
        ramp = raised_cosine(R)[:, None]
        x2[J - R: J] *= 1.0 + (g[None, :] - 1.0) * ramp
    return x2, g, R


# ------------------------------------------------------------------------------- release, sfizz

def release_time(x: np.ndarray, fs: int, release_onset: int, end: int) -> tuple[float, float]:
    """(sfizz ampeg_release, seconds the recording takes to fall 60 dB after release_onset).
    sfizz's release is exp(-9 t / T), i.e. -78.2 dB at t = T, so T = D * 78.2 / 60."""
    seg = x[release_onset: end + 1]
    if len(seg) < int(0.05 * fs):
        return 0.25, 0.0
    t, e = rms_envelope_db(seg, fs)
    e0 = float(np.max(e[: max(1, len(e) // 20)]))
    below = np.where(e <= e0 - 60.0)[0]
    if below.size:
        D = float(t[below[0]]) / fs
    else:
        fall = max(e0 - float(e[-1]), 6.0)
        D = (len(seg) / fs) * (60.0 / fall)
    T = D * 78.2 / 60.0
    return float(np.clip(T, 0.05, 4.0)), float(D)


_GAIN_CACHE: dict[int, float] = {}


def sfizz_gain_db(fs: int) -> float:
    """sfizz renders with a fixed headroom gain (about -11.5 dB); measure it once with a sine."""
    if fs in _GAIN_CACHE:
        return _GAIN_CACHE[fs]
    import tempfile
    import pysfizz
    with tempfile.TemporaryDirectory() as d:
        n = fs
        x = 0.5 * np.sin(2 * np.pi * 441.0 * np.arange(n) / fs)
        sf.write(os.path.join(d, 'cal.wav'), np.stack([x, x], 1), fs, subtype='PCM_16')
        with open(os.path.join(d, 'cal.sfz'), 'w') as f:
            f.write(f'<region> sample=cal.wav pitch_keycenter=69 lokey=69 hikey=69 amp_veltrack=0 '
                    f'loop_mode=loop_continuous loop_start=0 loop_end={n - 1}\n')
        syn = pysfizz.Synth(sample_rate=fs)
        syn.load_sfz_file(os.path.join(d, 'cal.sfz'))
        y = np.asarray(syn.render_note(69, 127, 1.0, 1.2))[0]
        g = 20 * np.log10(_rms(y[fs // 4: fs // 2]) / _rms(x[fs // 4: fs // 2]))
    _GAIN_CACHE[fs] = float(g)
    return float(g)


def render_sfizz(sfz_path: str, key: int, fs: int, note_on: float, total: float,
                 loop_start: int | None = None, L: int | None = None) -> dict | None:
    """Play the .sfz in sfizz (unity gain).  Returns dict(ok, audio (n, 2), early_db, late_db) or
    None without pysfizz.  ok: the loop sustains — the RMS over one whole loop length just after
    the join equals that over the last whole loop length before note-off (within 2 dB).  Whole
    loop lengths, because a long loop legitimately undulates several dB inside its period."""
    try:
        import pysfizz
    except ImportError:
        return None
    syn = pysfizz.Synth(sample_rate=fs)
    if not syn.load_sfz_file(os.path.abspath(sfz_path)):
        return dict(ok=False, error='sfizz could not load the file')
    y = np.asarray(syn.render_note(int(key), 100, float(note_on), float(total)), dtype=np.float64).T
    y *= 10 ** (-sfizz_gain_db(fs) / 20)
    if loop_start is not None and L is not None and note_on * fs > loop_start + 2 * L + int(0.1 * fs):
        a = loop_start + int(0.05 * fs)
        b = int(note_on * fs) - int(0.05 * fs)
        early = 20 * np.log10(_rms(y[a: a + L]))
        late = 20 * np.log10(_rms(y[b - L: b]))
    else:
        early = 20 * np.log10(_rms(y[int(0.3 * fs): int(0.6 * fs)]))
        late = 20 * np.log10(_rms(y[int((note_on - 0.4) * fs): int(note_on * fs)]))
    return dict(ok=bool(np.isfinite(early) and abs(late - early) < 2.0 and early > -60),
                early_db=float(early), late_db=float(late), peak=float(np.max(np.abs(y))), audio=y)


def _fade(y: np.ndarray, fs: int, ms_in: float = 2.0, ms_out: float = 20.0) -> np.ndarray:
    y = y.copy()
    a, b = int(ms_in * 1e-3 * fs), int(ms_out * 1e-3 * fs)
    if 1 < a < len(y):
        y[:a] *= raised_cosine(a)[:, None]
    if 1 < b < len(y):
        y[-b:] *= raised_cosine(b)[::-1][:, None]
    return y


# ------------------------------------------------------------------------------- the whole thing

@dataclass
class Replication:
    source: str
    fs: int
    basis: str
    method: str                  # 'bridge' (parameter-domain bridge) or 'splice' (join search + cross-fade)
    loop_seconds: float
    L: int
    K: int
    f0: list
    f0_used: float
    pitch_method: str
    keycenter: int
    tune_cents: float
    segments_s: dict
    body_s: list                 # [start, end] of the segment dctloop analysed, seconds into the file
    join_s: float                # J, seconds into the file
    attack_s: float              # J - onset: length of the recorded attack kept
    join_ncc: float              # waveform correlation of the recording with the loop's entry at J
    join_level_db: float         # recording level at J relative to the loop before the ramp
    attack_gain: list            # per-channel gain the ramp reaches at J
    ramp_samples: int
    xfade_samples: int
    morph_db: float
    loop_start: int
    loop_end: int                # inclusive
    loop_untouched: bool         # out[loop_start:loop_end+1] is exactly the dctloop loop (and no file gain)
    file_gain: float             # 1.0 unless the assembled file clipped and had to be scaled
    total_seconds: float
    release_sfz: float           # ampeg_release written
    release_fall60_s: float      # the recording's own fall time
    junction: dict
    continuity: dict             # per-harmonic and per-band steps across the join vs the recording's own
    bridge: dict | None          # bridge diagnostics (method='bridge')
    loop_metrics: dict
    sfizz: dict | None
    outputs: dict


def replicate_unaltered(path: str, out_dir: str | None = None, seconds: float = 0.5, *, method: str = 'bridge',
                        basis: str = 'dft', f0: float | None = None, use_hint: bool = True, lock: float = 1.5,
                        search_s: float = 0.5, bridge_s: float = 0.3, periods: float = 4.0, xfade_s: float = 0.01,
                        ramp_s: float = 0.08, morph: bool = False, morph_s: float = 0.25, morph_max_db: float = 6.0,
                        format: str = 'flac', preview: bool = True, sfizz: bool = True,
                        stem: str | None = None, verbose: bool = False) -> Replication:
    """Recorded note in, attack + untouched dctloop loop out.  See the module docstring.

    The loop is exactly what ``dctloop.loop_file`` would make of this file at ``seconds`` (same
    sustain detection, same pitch path, same basis and lock).

    method='bridge'  the join J is where the recording's harmonic amplitudes are closest to the
                     loop's (searched over ``search_s`` after attack_end + bridge_s), the attack's
                     level is ramped over the bridge, and every harmonic in the last ``bridge_s``
                     seconds is relaxed onto the loop's amplitude, frequency and phase (bridge.py).
                     Continuous by construction; a short cross-fade only carries the noise over.
    method='splice'  the join J is where the waveform best matches the loop's entry, then level
                     ramp, optional EQ morph and a cross-fade of ``xfade_s`` (the earlier method).
    """
    x, fs = load_audio(path)
    name = os.path.splitext(os.path.basename(path))[0]
    stem = stem or name
    seg = segment_note(x, fs)
    hint = f0 or note_from_name(name)

    # ---- the loop: exactly dctloop's
    a, b = find_body(x, fs)
    body = x[a:b]
    f0c, pinfo = f0_per_channel(body, fs, f0, hint, use_hint=use_hint, detail=True)
    good = f0c[np.isfinite(f0c)]
    if not good.size:
        raise ValueError(f'{name}: no pitch found')
    f0_used = float(np.exp(np.mean(np.log(good))))
    f0_list = [float(v) if np.isfinite(v) else f0_used for v in f0c]
    loop, linfo = loop_signal(body, fs, seconds, basis=basis, f0=f0_list, lock=lock)
    L = len(loop)
    seg_used = body[linfo['analysis_offset']: linfo['analysis_offset'] + linfo['N']]
    lm = measure(seg_used, fs, loop, f0_list)

    # ---- where to hand over, and how
    Wn = int(np.clip(4 * fs / max(f0_used, 20.0), 0.005 * fs, L // 2))
    binfo, tail, applied = None, None, 0.0
    if method == 'bridge':
        Bn = int(round(bridge_s * fs))
        lo = seg.attack_end + Bn
        hi = max(lo, min(lo + int(search_s * fs), seg.release_onset - L // 2, len(x) - Wn - 1))
        fj = find_join_bridge(x, fs, loop, f0_list, lo, hi)
        J = fj['J']
        lvl_j = 10 * np.log10(np.sum(x[J - Wn: J] ** 2) / (np.sum(loop[:Wn] ** 2) + 1e-30) + 1e-30)
        x2, g, R = fit_attack_level(x, seg.onset, J, loop, fs, f0_used, ramp_s=bridge_s)
        x2, binfo = bridge_attack(x2, fs, J, loop, f0_list, bridge_s=bridge_s)
        ncc_j = _ncc_at(x2, loop, J, Wn)
    elif method == 'splice':
        lo = seg.attack_end
        hi = min(seg.attack_end + int(search_s * fs), seg.release_onset - L // 2, len(x) - 1)
        fj = find_join(x, loop, fs, f0_used, lo, hi, periods=periods)
        J, lvl_j = fj['J'], fj['level_db']
        x2, g, R = fit_attack_level(x, seg.onset, J, loop, fs, f0_used, ramp_s=ramp_s)
        Mt = int(min(morph_s * fs, 0.7 * (J - seg.onset))) if morph else 0
        if Mt >= int(0.08 * fs):
            tail, applied = morph_tail(x2[J - Mt: J], loop, fs, max_db=morph_max_db)
        ncc_j = _ncc_at(x2, loop, J, Wn)
    else:
        raise ValueError(f"method must be 'bridge' or 'splice', got {method!r}")

    # ---- splice: recording -> short cross-fade into loop[L-X:] -> the loop itself
    out, ls, le, X = splice(x2, seg.onset, J, loop, fs, xfade_s=xfade_s, f0=f0_used, tail=tail)
    peak = float(np.max(np.abs(out)))
    file_gain = 0.999 / peak if peak > 0.999 else 1.0          # only ever needed if the loop itself clips
    untouched = bool(le - ls + 1 == L and file_gain == 1.0 and np.array_equal(out[ls: le + 1], loop))
    jm = junction_metrics(out, x2, seg.onset, ls, X, fs)
    cm = continuity_metrics(out, x, J, ls, fs, f0_list, loop=loop)
    T_rel, D60 = release_time(x, fs, seg.release_onset, seg.end)
    key, tune = key_and_tune(f0_used)

    outputs, sfz_info = {}, None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if file_gain != 1.0 and verbose:
            print(f'  [{name}] assembled file peaks at {peak:.3f}: whole file scaled by {20 * math.log10(file_gain):.2f} dB')
        ext = 'flac' if format == 'flac' else 'wav'
        p_audio = os.path.join(out_dir, f'{stem}.{ext}')
        sf.write(p_audio, out * file_gain, fs, subtype='PCM_24')
        outputs['audio'] = p_audio
        p_sfz = os.path.join(out_dir, f'{stem}.sfz')
        # the recording is `tune` cents above `key`; pitch_keycenter plays it at native pitch, so the
        # instrument needs tune=-offset to land on concert pitch (dctjoin.sfz.key_and_tune returns +offset)
        write_sfz(p_sfz, [dict(sample=f'{stem}.{ext}', keycenter=key, tune_cents=-tune, loop_start=ls, loop_end=le,
                               release=T_rel)], header=f'dctjoin (unaltered loop) replication of {name}')
        outputs['sfz'] = p_sfz
        note_on = (seg.release_onset - seg.onset) / fs
        if sfizz:
            # render at native pitch (tune=0) so the A/B preview compares like with like
            p_tmp = os.path.join(out_dir, f'{stem}__render.sfz')
            write_sfz(p_tmp, [dict(sample=f'{stem}.{ext}', keycenter=key, tune_cents=0.0, loop_start=ls, loop_end=le,
                                   release=T_rel)], header='temporary: native-pitch render for the preview')
            sfz_info = render_sfizz(p_tmp, key, fs, note_on, note_on + min(T_rel * 1.2, 3.0) + 0.1, loop_start=ls, L=L)
            os.remove(p_tmp)
            if sfz_info is not None and 'audio' in sfz_info:
                y = sfz_info.pop('audio')
                p_r = os.path.join(out_dir, f'{stem}_sfizz.wav')
                sf.write(p_r, np.clip(y, -1, 1), fs, subtype='PCM_16')
                outputs['sfizz_render'] = p_r
                if preview:
                    orig = _fade(x[seg.onset: seg.end + 1], fs)
                    rep = _fade(y[: min(len(y), len(orig) + int(0.5 * fs))], fs)
                    pv = np.concatenate([orig, np.zeros((int(0.4 * fs), x.shape[1])), rep], axis=0)
                    pk = float(np.max(np.abs(pv)))
                    if pk > 0.999:
                        pv *= 0.999 / pk
                    p_prev = os.path.join(out_dir, f'{stem}_preview.wav')
                    sf.write(p_prev, pv, fs, subtype='PCM_16')
                    outputs['preview'] = p_prev

    rep = Replication(source=path, fs=fs, basis=basis, method=method, loop_seconds=linfo['seconds'], L=L, K=linfo['K'],
                      f0=f0_list, f0_used=f0_used, pitch_method=pinfo.get('method', '?'), keycenter=key,
                      tune_cents=float(tune), segments_s=seg.seconds(fs), body_s=[a / fs, b / fs], join_s=J / fs,
                      attack_s=(J - seg.onset) / fs, join_ncc=float(ncc_j), join_level_db=float(lvl_j),
                      attack_gain=[float(v) for v in g], ramp_samples=R, xfade_samples=X, morph_db=float(applied),
                      loop_start=ls, loop_end=le, loop_untouched=untouched, file_gain=float(file_gain),
                      total_seconds=len(out) / fs,
                      release_sfz=T_rel, release_fall60_s=D60, junction=jm, continuity=cm, bridge=binfo,
                      loop_metrics=lm, sfizz=sfz_info, outputs=outputs)
    if out_dir:
        p_json = os.path.join(out_dir, f'{stem}.json')
        with open(p_json, 'w') as fh:
            json.dump(asdict(rep), fh, indent=1, default=float)
        outputs['json'] = p_json
    if verbose:
        print(summary_line(rep))
    return rep


def summary_line(r: Replication) -> str:
    j, m, s, c = r.junction, r.loop_metrics, r.sfizz, r.continuity
    sfz = 'sfizz ok' if (s and s.get('ok')) else ('sfizz FAIL' if s else 'sfizz n/a')
    br = f" bridge={r.bridge['bridge_samples'] / r.fs:.2f}s move={r.bridge['amp_move_db_weighted']:.1f}dB" if r.bridge else ''
    return (f'  [{os.path.splitext(os.path.basename(r.source))[0]}] {r.method}/{r.basis} loop={r.loop_seconds:.3f}s '
            f'attack={r.attack_s:.3f}s ncc={r.join_ncc:.2f} lvl={r.join_level_db:+.1f}dB{br} '
            f'| join: harm {c["harm_step_db_wmean"]:.2f}dB/{c["harm_phase_err_deg_wmean"]:.0f}deg bands {c["band_step_db_mean"]:.1f}dB '
            f'(loop own {c.get("loop_harm_step_db_wmean", 0):.2f}dB/{c.get("loop_harm_phase_err_deg_wmean", 0):.0f}deg, '
            f'rec {c["rec_harm_step_db_wmean"]:.2f}dB/{c["rec_harm_phase_err_deg_wmean"]:.0f}deg bands {c["rec_band_step_db_mean"]:.1f}dB) '
            f'tail ncc={c.get("tail_ncc", float("nan")):.3f} resid={c.get("tail_residual_db", float("nan")):.0f}dB '
            f'| untouched={"yes" if r.loop_untouched else "NO"} | {sfz}')
