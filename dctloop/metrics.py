"""dctloop.metrics — is the loop seamless, stable and does it still sound like the input?

* ``seam_metrics``   spectral flux at the seam (and at the palindrome mid-point) relative to
                     the median flux of the loop: ~1 means the join is no more eventful than
                     any other instant; ``p95_flux_ratio`` gives the scale of normal variation.
* ``harmonic_am``    the "wah": energy on the grid bins next to each harmonic beats with it at
                     exactly 1/L Hz.  Reports the worst peak-to-peak swell in dB (0 = none).
* ``spectrum_match`` third-octave long-term spectrum of the loop against the analysed segment,
                     per channel, plus the level of the mono sum (which is sensitive to how the
                     inter-channel phases were frozen).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import stft, welch


def seam_metrics(loop: np.ndarray, fs: int) -> dict:
    L = len(loop)
    mono = loop.mean(axis=1) if loop.ndim == 2 else loop
    n = 2048 if L >= 8192 else max(128, 1 << int(math.log2(max(L // 4, 128))))
    hop = n // 8
    y = np.tile(mono, 4)
    _, t, Z = stft(y, fs, window='hann', nperseg=n, noverlap=n - hop, boundary=None, padded=False)
    mag = np.abs(Z)
    flux = np.linalg.norm(np.diff(mag, axis=1), axis=0) / (np.linalg.norm(mag[:, 1:], axis=0) + 1e-12)
    centres = t[1:] * fs
    med = float(np.median(flux)) + 1e-12

    def ratio(points):
        sel = np.zeros(len(centres), bool)
        for p in points:
            sel |= np.abs(centres - p) <= n / 2
        return float(flux[sel].max() / med) if sel.any() else float('nan')

    return dict(seam_flux_ratio=ratio([L, 2 * L, 3 * L]),
                mid_flux_ratio=ratio([1.5 * L, 2.5 * L]),
                p95_flux_ratio=float(np.percentile(flux, 95) / med),
                seam_step=float(np.max(np.abs(loop[0] - loop[-1])) / (np.max(np.abs(loop)) + 1e-12)))


def harmonic_am(loop: np.ndarray, fs: int, f0: float | list, nharm: int = 10) -> dict:
    """Energy on the grid bins next to harmonic h (h * K +- 1) relative to the harmonic, and the
    resulting peak-to-peak amplitude modulation at 1/L Hz; the worst over channels and over the
    harmonics that carry at least 2 % of the peak amplitude."""
    if loop.ndim == 1:
        loop = loop[:, None]
    L = len(loop)
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    worst, rows = 0.0, []
    for c in range(loop.shape[1]):
        Y = np.abs(np.fft.rfft(loop[:, c])) * 2 / L
        kf = f0s[min(c, len(f0s) - 1)] * L / fs
        for h in range(1, nharm + 1):
            m = int(round(h * kf))
            if m + 1 >= len(Y) or m < 2:
                break
            hv, sv = Y[m], Y[m - 1] + Y[m + 1]
            if hv < 1e-6:
                continue
            side_db = 20 * math.log10(sv / hv + 1e-12)
            am_db = 20 * math.log10((hv + sv) / (hv - sv)) if hv > sv else float('inf')
            rows.append((c, h, round(side_db, 1)))
            if hv > 0.02 * Y.max():
                worst = max(worst, am_db)
    return dict(max_harmonic_am_db=float(worst), side_db=rows)


def spectrum_match(seg: np.ndarray, fs: int, loop: np.ndarray, lo: float = 60.0,
                   floor_db: float = 60.0) -> dict:
    """Third-octave LTAS of the loop (tiled to the segment's length) vs the segment, per
    channel.  Bands more than ``floor_db`` below the loudest band of the reference are ignored
    (noise floors, dither)."""
    y = np.tile(loop, (int(math.ceil(len(seg) / len(loop))), 1))[:len(seg)]
    nper = min(8192, len(seg))
    hi = min(16000.0, 0.45 * fs)
    edges = lo * 2.0 ** (np.arange(0, int(3 * math.log2(hi / lo)) + 2) / 3.0)
    per_chan = []
    for c in range(seg.shape[1]):
        f, P1 = welch(seg[:, c], fs, nperseg=nper)
        _, P2 = welch(y[:, c], fs, nperseg=nper)
        e1, e2 = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (f >= a) & (f < b)
            if sel.sum() < 2:
                continue
            e1.append(P1[sel].sum() + 1e-20)
            e2.append(P2[sel].sum() + 1e-20)
        e1, e2 = np.array(e1), np.array(e2)
        d = 10 * np.log10(e2 / e1)
        d[e1 < e1.max() * 10 ** (-floor_db / 10)] = 0.0
        per_chan.append(d)
    d = np.mean(np.abs(np.array(per_chan)), axis=0)
    mono_db = 10 * np.log10(np.mean(y.mean(axis=1) ** 2) / (np.mean(seg.mean(axis=1) ** 2) + 1e-20) + 1e-20)
    return dict(ltas_mean_abs_db=float(np.mean(d)), ltas_max_abs_db=float(np.max(d)), mono_db=float(mono_db),
                ltas_bands_db=[[round(float(v), 2) for v in ch] for ch in per_chan])


def measure(seg: np.ndarray, fs: int, loop: np.ndarray, f0: float | list | None = None) -> dict:
    """All of the above in one dict."""
    out = seam_metrics(loop, fs)
    if f0 is not None:
        out.update(harmonic_am(loop, fs, f0))
    out.update(spectrum_match(seg, fs, loop))
    return out
