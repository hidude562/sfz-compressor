"""dctjoin.pipeline — a recorded note in, a playable replication out.

    recording -> segment (attack end, release onset) -> pitch -> dctloop loop from the body
              -> join (align, level, morph, splice) at the best of a few join points
              -> <name>.flac (attack + loop) + <name>.sfz + preview + sfizz render + json
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf

from dctloop import f0_per_channel, load_audio, loop_signal, measure, note_from_name

from .join import join, raised_cosine
from .segment import segment_note
from .sfz import key_and_tune, write_sfz


@dataclass
class Replication:
    source: str
    fs: int
    basis: str
    f0: list
    f0_used: float
    pitch_method: str
    keycenter: int
    tune_cents: float
    segments_s: dict            # onset / peak / attack_end / release_onset / end in seconds
    join_s: float               # chosen join, seconds into the file
    loop_seconds: float
    L: int
    K: int
    loop_start: int             # SFZ loop points, samples into the written file
    loop_end: int               # inclusive
    total_seconds: float        # length of the written file
    alignment: str
    gain: list
    xfade_samples: int
    morph_db: float
    junction: dict
    loop_metrics: dict
    candidates: list            # every join point tried: join_s, score, alignment, loop_seconds
    sfizz: dict | None
    outputs: dict


def _fade_out(y: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(y))
    if n > 1:
        y = y.copy()
        y[-n:] *= raised_cosine(n)[::-1][:, None]
    return y


def make_preview(x: np.ndarray, onset: int, out: np.ndarray, loop_start: int, fs: int,
                 seconds: float = 4.0, gap: float = 0.4) -> np.ndarray:
    """[original note] gap [replication: attack then the loop repeating] — same length each."""
    n = int(seconds * fs)
    orig = _fade_out(x[onset: onset + n], int(0.02 * fs))
    loop = out[loop_start:]
    reps = int(math.ceil(max(0, n - loop_start) / len(loop))) + 1
    rep = _fade_out(np.concatenate([out[:loop_start], np.tile(loop, (reps, 1))], axis=0)[:len(orig)], int(0.02 * fs))
    return np.concatenate([orig, np.zeros((int(gap * fs), x.shape[1])), rep], axis=0)


def render_sfizz(sfz_path: str, key: int, fs: int, note_on: float = 2.0, seconds: float = 3.0) -> dict | None:
    """Load the SFZ in sfizz and play the note: returns the render and a hold check (does the
    loop actually sustain?), or None if pysfizz is not installed."""
    try:
        import pysfizz
    except ImportError:
        return None
    syn = pysfizz.Synth(sample_rate=fs)
    if not syn.load_sfz_file(sfz_path):
        return dict(ok=False, error='sfizz could not load the file')
    y = np.asarray(syn.render_note(key, 100, note_on, seconds), dtype=np.float64).T
    def rms_db(a):
        return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12)
    early = rms_db(y[int(0.3 * fs): int(0.6 * fs)])
    late = rms_db(y[int((note_on - 0.4) * fs): int(note_on * fs)])
    return dict(ok=bool(np.isfinite(early) and abs(late - early) < 3.0 and early > -60),
                early_db=float(early), late_db=float(late), peak=float(np.max(np.abs(y))), audio=y)


def replicate_file(path: str, out_dir: str | None = None, seconds: float = 0.25, *, basis: str = 'dft',
                   f0: float | None = None, use_hint: bool = True, lock: float = 1.5, align: str = 'best',
                   xfade_s: float = 0.01, morph: bool = True, morph_s: float = 0.25, morph_max_db: float = 6.0,
                   join_offsets: tuple = (0.0, 0.15, 0.3), format: str = 'flac', preview: bool = True,
                   preview_seconds: float = 4.0, sfizz: bool = True, release: float = 0.25,
                   stem: str | None = None, verbose: bool = False) -> Replication:
    """Replicate one recorded note as attack + seamless loop.  See the module docstring."""
    x, fs = load_audio(path)
    name = os.path.splitext(os.path.basename(path))[0]
    stem = stem or name
    seg = segment_note(x, fs)
    hint = f0 or note_from_name(name)
    f0c, pinfo = f0_per_channel(x[seg.attack_end:seg.release_onset], fs, f0, hint, use_hint=use_hint, detail=True)
    good = f0c[np.isfinite(f0c)]
    if not good.size:
        raise ValueError(f'{name}: no pitch found')
    f0_used = float(np.exp(np.mean(np.log(good))))
    f0_list = [float(v) if np.isfinite(v) else f0_used for v in f0c]

    best, cands = None, []
    for off in join_offsets:
        J = seg.attack_end + int(off * fs)
        body = x[J:seg.release_onset]
        if len(body) < 16 * fs / f0_used or len(body) < 0.3 * fs:
            continue
        loop, linfo = loop_signal(body, fs, seconds, basis=basis, f0=f0_list, lock=lock)
        out, jinfo = join(x, fs, seg.onset, J, loop, f0_used, basis=basis, align=align, xfade_s=xfade_s,
                          morph=morph, morph_s=morph_s, morph_max_db=morph_max_db)
        sc = jinfo['junction']['score']
        cands.append(dict(join_s=J / fs, score=round(sc, 3), alignment=jinfo['alignment'],
                          loop_seconds=round(linfo['seconds'], 4), dip_db=round(jinfo['junction']['dip_db'], 2),
                          transient_db=round(jinfo['junction']['transient_db'], 2)))
        if best is None or sc < best[0]:
            seg_used = body[linfo['analysis_offset']: linfo['analysis_offset'] + linfo['N']]
            best = (sc, J, loop, linfo, out, jinfo, seg_used)
    if best is None:
        raise ValueError(f'{name}: not enough steady material after the attack to loop')
    _, J, loop, linfo, out, jinfo, seg_used = best
    lm = measure(seg_used, fs, loop, f0_list)          # the loop as dctloop made it, before the join gain
    key, tune = key_and_tune(f0_used)

    outputs, sfz_info = {}, None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        peak = float(np.max(np.abs(out)))
        if peak > 0.999:
            out = out * (0.999 / peak)
        ext = 'flac' if format == 'flac' else 'wav'
        p_audio = os.path.join(out_dir, f'{stem}.{ext}')
        sf.write(p_audio, out, fs, subtype='PCM_24')
        outputs['audio'] = p_audio
        p_sfz = os.path.join(out_dir, f'{stem}.sfz')
        write_sfz(p_sfz, [dict(sample=f'{stem}.{ext}', keycenter=key, tune_cents=tune, loop_start=jinfo['loop_start'],
                               loop_end=jinfo['loop_end'], release=release)], header=f'dctjoin replication of {name}')
        outputs['sfz'] = p_sfz
        if preview:
            pv = make_preview(x, seg.onset, out, jinfo['loop_start'], fs, seconds=preview_seconds)
            pk = float(np.max(np.abs(pv)))
            if pk > 0.999:
                pv *= 0.999 / pk
            p_prev = os.path.join(out_dir, f'{stem}_preview.wav')
            sf.write(p_prev, pv, fs, subtype='PCM_16')
            outputs['preview'] = p_prev
        if sfizz:
            sfz_info = render_sfizz(p_sfz, key, fs)
            if sfz_info is not None:
                y = sfz_info.pop('audio', None)
                if y is not None:
                    p_r = os.path.join(out_dir, f'{stem}_sfizz.wav')
                    pk = float(np.max(np.abs(y)))
                    sf.write(p_r, y * (0.9 / pk) if pk > 0 else y, fs, subtype='PCM_16')
                    outputs['sfizz_render'] = p_r

    rep = Replication(source=path, fs=fs, basis=basis, f0=f0_list, f0_used=f0_used, pitch_method=pinfo.get('method', '?'),
                      keycenter=key, tune_cents=float(tune), segments_s=seg.seconds(fs), join_s=J / fs,
                      loop_seconds=linfo['seconds'], L=linfo['L'], K=linfo['K'], loop_start=jinfo['loop_start'],
                      loop_end=jinfo['loop_end'], total_seconds=len(out) / fs, alignment=jinfo['alignment'],
                      gain=jinfo['gain'], xfade_samples=jinfo['xfade'], morph_db=float(jinfo['morph_db']),
                      junction=jinfo['junction'], loop_metrics=lm, candidates=cands, sfizz=sfz_info, outputs=outputs)
    if out_dir:
        p_json = os.path.join(out_dir, f'{stem}.json')
        with open(p_json, 'w') as fh:
            json.dump(asdict(rep), fh, indent=1, default=float)
        outputs['json'] = p_json
    if verbose:
        print(summary_line(rep))
    return rep


def summary_line(r: Replication) -> str:
    j, m = r.junction, r.loop_metrics
    s = r.sfizz
    sfz = 'sfizz ok' if (s and s.get('ok')) else ('sfizz FAIL' if s else 'sfizz n/a')
    return (f'  [{os.path.splitext(os.path.basename(r.source))[0]}] {r.basis} f0={r.f0_used:.2f}Hz/{r.pitch_method} '
            f'key={r.keycenter}{r.tune_cents:+.0f}c attack={r.segments_s["attack_end"] - r.segments_s["onset"]:.3f}s '
            f'join@{r.join_s - r.segments_s["onset"]:.3f}s loop={r.loop_seconds:.3f}s ({r.K}p) '
            f'align={r.alignment} xfade={r.xfade_samples} morph={r.morph_db:.1f}dB '
            f'dip={j["dip_db"]:+.2f}dB transient={j["transient_db"]:+.1f}dB score={j["score"]:.2f} '
            f'| seam×{m["seam_flux_ratio"]:.2f} wah={m.get("max_harmonic_am_db", 0):.1f}dB ltas={m["ltas_mean_abs_db"]:.2f}dB '
            f'| file={r.total_seconds:.2f}s {sfz}')
