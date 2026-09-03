"""dctloop.pipeline — from a recorded note on disk to a loop file: load, find the sustain,
pitch, loop, measure, write (loop, A/B preview, JSON report)."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf

from .core import _rms, loop_signal
from .metrics import measure
from .pitch import note_from_name


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """(samples as float64 (N, C), sample rate)."""
    x, fs = sf.read(path, dtype='float64', always_2d=True)
    return x, int(fs)


def rms_envelope(x: np.ndarray, fs: int, win: float = 0.02, hop: float = 0.005):
    """(frame centres in samples, RMS in dB) of the channel mean."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    w, h = max(2, int(win * fs)), max(1, int(hop * fs))
    if len(mono) < w:
        return np.array([len(mono) // 2]), np.array([20 * np.log10(_rms(mono))])
    frames = np.lib.stride_tricks.sliding_window_view(mono, w)[::h]
    e = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    return np.arange(len(frames)) * h + w // 2, 20 * np.log10(e)


def find_body(x: np.ndarray, fs: int, within_db: float = 12.0, guard: float = 0.15) -> tuple[int, int]:
    """Sample range of the sustained body: the longest run of the smoothed RMS envelope that
    stays within ``within_db`` of its peak, trimmed by ``guard`` seconds at both ends."""
    t, e = rms_envelope(x, fs)
    if len(e) >= 9:
        e = np.convolve(e, np.ones(9) / 9, mode='same')
    ok = np.append(e >= e.max() - within_db, False)
    best, start = (0, 0), None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    a, b = int(t[best[0]]), int(t[best[1] - 1])
    g = int(guard * fs)
    a, b = a + g, b - g
    if b - a < int(0.1 * fs):  # degenerate: fall back to the middle half of the file
        a, b = len(x) // 4, 3 * len(x) // 4
    return a, b


def _fade(x: np.ndarray, fs: int, ms: float = 10.0) -> np.ndarray:
    n = min(len(x) // 2, int(ms * 1e-3 * fs))
    if n < 1:
        return x
    y = x.copy()
    ramp = np.linspace(0, 1, n)[:, None]
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def make_preview(seg: np.ndarray, loop: np.ndarray, fs: int, orig_seconds: float = 2.0,
                 loop_seconds: float = 4.0, gap: float = 0.4) -> np.ndarray:
    """[original excerpt] gap [loop repeated] — for A/B listening."""
    o = _fade(seg[:int(orig_seconds * fs)], fs)
    reps = int(math.ceil(loop_seconds * fs / len(loop)))
    l = _fade(np.tile(loop, (reps, 1))[:int(loop_seconds * fs)], fs)
    return np.concatenate([o, np.zeros((int(gap * fs), seg.shape[1])), l], axis=0)


@dataclass
class LoopResult:
    """What ``loop_file`` returns (and writes as ``<stem>.json``)."""
    source: str
    fs: int
    basis: str
    f0_hint: float | None
    f0: list                 # per channel, Hz
    f0_used: float           # geometric mean over channels: the loop fit uses this
    detune_cents: float      # last channel vs first
    loop_seconds: float
    L: int                   # loop length in samples
    K: int                   # fundamental periods per loop
    grid_cents: float        # pitch error of the grid vs f0_used
    seg_start: float         # analysed segment, seconds into the file
    seg_seconds: float
    info: dict               # analysis details (frames, grid_offset, gain, lock_width ...)
    metrics: dict            # seam_metrics + harmonic_am + spectrum_match
    outputs: dict            # paths written


def loop_file(path: str, out_dir: str | None = None, seconds: float = 1.5, *, basis: str = 'dct',
              f0: float | None = None, use_hint: bool = True, fit: bool = True,
              periods: int | None = None, lock: float = 1.5, phase: str = 'orig', start: float | None = None,
              dur: float | None = None, preview: bool = True, stem: str | None = None,
              seed: int = 0, verbose: bool = False) -> LoopResult:
    """Loop a recorded note.  Picks the sustain automatically (or ``start``/``dur`` seconds),
    takes the note name in the file name as the pitch (verified against the spectrum, else
    pYIN — see pitch.f0_per_channel; ``use_hint=False`` forces pYIN), builds the loop,
    measures it, and — if ``out_dir`` is given — writes ``<stem>_loop.wav`` (24-bit),
    ``<stem>_preview.wav`` (original, gap, loop repeated) and ``<stem>.json``."""
    x, fs = load_audio(path)
    name = os.path.splitext(os.path.basename(path))[0]
    stem = stem or name

    if start is not None:
        a = int(start * fs)
        b = int((start + dur) * fs) if dur else len(x)
    else:
        a, b = find_body(x, fs)
        if dur:
            mid = (a + b) // 2
            a, b = max(0, mid - int(dur * fs / 2)), min(len(x), mid + int(dur * fs / 2))
    seg = x[a:b]
    hint = f0 or note_from_name(name)

    lp, info = loop_signal(seg, fs, seconds, basis=basis, f0=f0, hint=hint, use_hint=use_hint,
                           fit=fit, periods=periods, lock=lock, phase=phase, seed=seed)
    if info['shortened'] and verbose:
        print(f'  [{name}] only {len(seg) / fs:.2f}s of sustain: loop shortened to {info["seconds"]:.3f}s')
    seg_used = seg[info['analysis_offset']:info['analysis_offset'] + info['N']]
    f0c = info['f0']
    met = measure(seg_used, fs, lp, [v if np.isfinite(v) else info['f0_used'] for v in f0c]
                  if info['f0_used'] > 0 else None)

    outputs = {}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        peak = float(np.max(np.abs(lp)))
        if peak > 0.999:
            lp = lp * (0.999 / peak)
        p_loop = os.path.join(out_dir, f'{stem}_loop.wav')
        sf.write(p_loop, lp, fs, subtype='PCM_24')
        outputs['loop'] = p_loop
        if preview:
            pv = make_preview(seg_used, lp, fs)
            pk = float(np.max(np.abs(pv)))
            if pk > 0.999:
                pv *= 0.999 / pk
            p_prev = os.path.join(out_dir, f'{stem}_preview.wav')
            sf.write(p_prev, pv, fs, subtype='PCM_16')
            outputs['preview'] = p_prev

    res = LoopResult(source=path, fs=fs, basis=basis, f0_hint=hint, f0=f0c, f0_used=info['f0_used'],
                     detune_cents=info['detune_cents'], loop_seconds=info['seconds'], L=info['L'],
                     K=info['K'], grid_cents=info['grid_cents'],
                     seg_start=(a + info['analysis_offset']) / fs, seg_seconds=info['N'] / fs,
                     info=info, metrics=met, outputs=outputs)
    if out_dir:
        p_json = os.path.join(out_dir, f'{stem}.json')
        with open(p_json, 'w') as fh:
            json.dump(asdict(res), fh, indent=1)
        outputs['json'] = p_json
    if verbose:
        print(summary_line(res))
    return res


def summary_line(r: LoopResult) -> str:
    m = r.metrics
    pm = r.info.get('pitch', {}).get('method', '?')
    return (f'  [{os.path.splitext(os.path.basename(r.source))[0]}] {r.basis} f0={r.f0_used:.2f}Hz/{pm} '
            f'(L/R {r.detune_cents:+.1f}c) loop={r.loop_seconds:.3f}s ({r.K} periods, grid {r.grid_cents:+.2f}c) '
            f'seg={r.seg_seconds:.2f}s frames={r.info.get("frames")} '
            f'seam×{m["seam_flux_ratio"]:.2f} mid×{m["mid_flux_ratio"]:.2f} p95×{m["p95_flux_ratio"]:.2f} '
            f'ltas={m["ltas_mean_abs_db"]:.2f}/{m["ltas_max_abs_db"]:.2f}dB mono={m["mono_db"]:+.1f}dB '
            f'wah={m.get("max_harmonic_am_db", 0):.1f}dB gain={r.info["gain"]:.2f}')


# backwards-compatible name
process = loop_file
