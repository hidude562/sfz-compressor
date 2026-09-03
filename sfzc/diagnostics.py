"""Loop diagnostics (report section 5): seam flux, sample continuity, DC, integer-cycle check."""
from __future__ import annotations

import numpy as np

from . import dsp


def diagnose_loop(loop: np.ndarray, sr: int, locked_freqs: np.ndarray | None = None,
                  orig_freqs: np.ndarray | None = None, tile: int = 8) -> dict:
    """loop: (N, C) or (N,) one period of the loop."""
    from .metric import _novelty

    if loop.ndim == 1:
        loop = loop[:, None]
    N = len(loop)
    m = dsp.to_mono(np.tile(loop, (tile, 1)))
    hop = 256
    nov = _novelty(m, sr, hop)
    k = max(3, int(round(0.05 * sr / hop)) | 1)
    hp = np.maximum(nov - dsp.smooth(nov, k), 0)
    frames = np.arange(len(hp))
    per = N / hop
    seam_frames = [int(round(j * per)) for j in range(1, tile) if int(round(j * per)) < len(hp) - 1]
    seam_vals = np.array([hp[max(0, s - 1): s + 2].max() for s in seam_frames]) if seam_frames else np.zeros(0)
    interior = np.array([hp[i] for i in frames if all(abs(i - s) > 2 for s in seam_frames)])
    seam_db = 20 * np.log10((seam_vals.mean() + 1e-9) / (np.percentile(interior, 90) + 1e-9)) if seam_vals.size else 0.0
    seam_pct = float(np.mean(interior <= seam_vals.mean())) * 100 if seam_vals.size else 0.0
    d1 = np.abs(np.diff(loop, axis=0)).mean(axis=1)
    step = np.abs(loop[0] - loop[-1]).mean()
    d2 = np.abs(np.diff(m, 2))
    seam_d2 = max(float(d2[j * N - 3: j * N + 2].max()) for j in range(1, tile) if j * N + 2 < len(d2))
    out = dict(N=N, loop_s=N / sr, seam_flux_db=float(seam_db), seam_flux_percentile=float(seam_pct),
               seam_step=float(step), typical_step_p99=float(np.percentile(d1, 99)),
               seam_d2=seam_d2, interior_d2_p999=float(np.percentile(d2, 99.9)),
               dc_offset_rel=float(abs(loop.mean()) / (np.sqrt((loop ** 2).mean()) + 1e-12)))
    if locked_freqs is not None and len(locked_freqs):
        cyc = np.asarray(locked_freqs, float) * N / sr
        out["max_cycle_frac_error"] = float(np.max(np.abs(cyc - np.round(cyc))))
        if orig_freqs is not None:
            of = np.asarray(orig_freqs, float)
            cents = 1200 * np.log2(np.maximum(locked_freqs, 1e-3) / np.maximum(of, 1e-3))
            out["detune_cents_max"] = float(np.max(np.abs(cents)))
            out["detune_cents_rms"] = float(np.sqrt(np.mean(cents ** 2)))
    return out


def format_report(d: dict) -> str:
    lines = [f"loop {d['N']} samples ({d['loop_s'] * 1000:.1f} ms)",
             f"seam spectral flux: {d['seam_flux_db']:+.1f} dB vs interior p90 (seam at the {d['seam_flux_percentile']:.0f}th percentile of interior frames)",
             f"sample continuity: |y[0]-y[N-1]| = {d['seam_step']:.5f} (typical p99 step {d['typical_step_p99']:.5f}); "
             f"|d2| at seam {d['seam_d2']:.5f} vs interior p99.9 {d['interior_d2_p999']:.5f}",
             f"DC offset: {d['dc_offset_rel'] * 100:.3f} % of RMS"]
    if "max_cycle_frac_error" in d:
        lines.append(f"integer cycles: max fractional error {d['max_cycle_frac_error']:.2e}")
    if "detune_cents_max" in d:
        lines.append(f"partial detuning: max {d['detune_cents_max']:.2f} cents, rms {d['detune_cents_rms']:.2f} cents")
    return "\n".join(lines)
