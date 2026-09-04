"""dctjoin.bridge — a parameter-domain bridge from the recorded attack onto an untouched loop.

The splice in ``unaltered.py`` can only pick *where* to join; if the recording never sounds like
the loop's entry, the join steps in timbre.  The bridge makes the join continuous by construction.

Per channel and per harmonic band h (the frequencies within half a harmonic spacing of h*f0):

* ``harmonic_tracks``     heterodyne the recording over the bridge region, the last B seconds
                          before the join J: a complex amplitude every ``hop`` samples, from a
                          Hann window of a few periods around h*f0 (neighbouring harmonics are
                          rejected, vibrato and the beating of a section's chorus are captured).
                          Gives the band's amplitude A_rec(t) and unwrapped total phase p_rec(t).
* ``loop_band_targets``   what the *loop* does in that band just before its sample 0 (which will
                          sit at J): the complex sum of all of the loop's grid partials in the
                          band, evaluated at the same instants.  One partial for a locked
                          harmonic; several, beating, for a section's chorus.  Gives A_tgt(t) and
                          p_tgt(t), the trajectory the recording has to arrive on.
* ``bridge_attack``       re-synthesise the band twice — from the measured (A_rec, p_rec) and from
                          the relaxed (A_m, p_m) — and add the difference to the recording:

                              A_m = exp((1-s) ln A_rec + s ln A_tgt)
                              p_m = p_rec + s * D,   D = p_tgt - p_rec  (unwrapped, |D(J)| <= pi)

                          with s(t) a raised cosine from 0 at J-B to 1 at J.  At J-B the two
                          syntheses are identical, so the correction is exactly zero and the
                          recording is untouched; at J the band has exactly the loop's amplitude,
                          phase and instantaneous frequency (s' = 0 there).  Everything the model
                          does not describe — noise, the transient, fluctuations faster than the
                          analysis window — passes through unchanged.  Vibrato and tremolo die
                          away into the loop's own motion instead of stopping dead.  The phase
                          offset D(J), at most pi, spread over B seconds is a momentary frequency
                          deviation of at most pi/(4B) Hz: 2.6 Hz for B = 0.3 s.
* ``continuity_metrics``  did it work?  Per-harmonic amplitude step and phase error across the
                          join, third-octave band steps, all against the recording's own.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal.windows import get_window

from .join import raised_cosine


def _rms(a, axis=None):
    return np.sqrt(np.mean(np.square(a), axis=axis)) + 1e-12


# ------------------------------------------------------------------------------- the loop's harmonics

def loop_harmonics(loop: np.ndarray, fs: int, f0: list | float, nh: int, search: int = 3) -> dict:
    """The strongest grid partial per channel c and harmonic h (1..nh), within +-``search`` bins
    of round(h * f0_c * L / fs): amplitude A[h-1, c], frequency f[h-1, c] (Hz), phase phi[h-1, c]
    at loop sample 0.  (Single-partial view; ``loop_band_targets`` is the full one.)"""
    if loop.ndim == 1:
        loop = loop[:, None]
    L, C = loop.shape
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    Y = np.fft.rfft(loop, axis=0)
    A = np.zeros((nh, C))
    f = np.zeros((nh, C))
    phi = np.zeros((nh, C))
    for c in range(C):
        fc = f0s[min(c, len(f0s) - 1)]
        for h in range(1, nh + 1):
            m0 = int(round(h * fc * L / fs))
            lo, hi = max(1, m0 - search), min(len(Y) - 1, m0 + search + 1)
            if hi <= lo:
                continue
            m = lo + int(np.argmax(np.abs(Y[lo:hi, c])))
            A[h - 1, c] = 2.0 * abs(Y[m, c]) / L
            f[h - 1, c] = m * fs / L
            phi[h - 1, c] = float(np.angle(Y[m, c]))
    return dict(A=A, f=f, phi=phi, L=L)


def loop_band_targets(loop: np.ndarray, fs: int, f0: list | float, nh: int, n_rel: np.ndarray,
                      rel_db: float = -50.0) -> tuple[np.ndarray, list]:
    """Complex value T[k, h-1, c] of the loop's band-h content at loop sample n_rel[k] (negative
    before sample 0; the loop is periodic), summing every grid partial within half a harmonic
    spacing of h * f0_c that is within ``rel_db`` of the band's strongest.  Also returns the bins
    used per (h, c)."""
    if loop.ndim == 1:
        loop = loop[:, None]
    L, C = loop.shape
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    Y = np.fft.rfft(loop, axis=0)
    amp = 2.0 * np.abs(Y) / L
    ph = np.angle(Y)
    fbin = np.arange(len(Y)) * fs / L
    T = np.zeros((len(n_rel), nh, C), dtype=complex)
    used = []
    for c in range(C):
        fc = f0s[min(c, len(f0s) - 1)]
        for h in range(1, nh + 1):
            sel = np.where((fbin >= (h - 0.5) * fc) & (fbin < (h + 0.5) * fc) & (np.arange(len(Y)) >= 1))[0]
            if not sel.size:
                used.append([])
                continue
            a = amp[sel, c]
            keep = sel[a >= a.max() * 10 ** (rel_db / 20)]
            used.append([int(m) for m in keep])
            # sum of partials  a_m exp(i (2 pi m n / L + phi_m))  at every n in n_rel
            T[:, h - 1, c] = np.sum(amp[keep, c][None, :] * np.exp(1j * (2 * np.pi * np.outer(n_rel, keep) / L + ph[keep, c][None, :])), axis=1)
    return T, used


# ------------------------------------------------------------------------------- the recording's harmonics

def harmonic_tracks(x: np.ndarray, fs: int, f0: list | float, nh: int, centres: np.ndarray,
                    periods: float = 6.0) -> dict:
    """Heterodyne analysis of ``x`` (N, C) at frame ``centres`` (samples): per band h and channel
    c, amplitude amp[k, h, c], unwrapped total phase phase[k, h, c] (the phase of cos at the frame
    centre, continuous across frames) and instantaneous frequency freq[k, h, c] (Hz, from the
    baseband phase increments, so never aliased)."""
    if x.ndim == 1:
        x = x[:, None]
    N, C = x.shape
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    K = len(centres)
    amp = np.zeros((K, nh, C))
    phase = np.zeros((K, nh, C))
    freq = np.zeros((K, nh, C))
    for c in range(C):
        fc = f0s[min(c, len(f0s) - 1)]
        W = int(round(periods * fs / fc)) | 1                       # odd: symmetric about the centre
        half = W // 2
        w = get_window('hann', W, fftbins=False)
        wsum = w.sum()
        n_rel = np.arange(-half, half + 1)
        idx = centres[:, None] + n_rel[None, :]
        valid = (idx >= 0) & (idx < N)
        frames = np.where(valid, x[np.clip(idx, 0, N - 1), c], 0.0) * w[None, :]
        for h in range(1, nh + 1):
            fh = h * fc
            # demodulate with the carrier referred to absolute time: z is the slowly varying
            # complex envelope (only the deviation from h*f0 is left), safe to unwrap
            z = np.sum(frames * np.exp(-2j * np.pi * fh * (n_rel / fs))[None, :], axis=1) * (2.0 / wsum)
            z = z * np.exp(-2j * np.pi * fh * (centres / fs))
            amp[:, h - 1, c] = np.abs(z)
            phase[:, h - 1, c] = np.unwrap(np.angle(z)) + 2 * np.pi * fh * (centres / fs)
            if K > 1:
                dph = np.angle(z[1:] * np.conj(z[:-1]))                  # baseband increment per hop
                dt = np.diff(centres) / fs
                f_int = fh + dph / (2 * np.pi * dt)
                freq[:, h - 1, c] = np.concatenate([[f_int[0]], 0.5 * (f_int[1:] + f_int[:-1]), [f_int[-1]]])
            else:
                freq[:, h - 1, c] = fh
    return dict(amp=amp, phase=phase, freq=freq, centres=centres)


def _synth(amp: np.ndarray, phase: np.ndarray, centres: np.ndarray, n0: int, n1: int) -> np.ndarray:
    """Sum of bands over samples [n0, n1) from frame-rate amplitude and total phase (K, nh, C),
    both linearly interpolated between frame centres (hop ~ one period, so exact for a steady
    partial)."""
    n = np.arange(n0, n1)
    K, nh, C = amp.shape
    out = np.zeros((n1 - n0, C))
    for c in range(C):
        for h in range(nh):
            a = np.interp(n, centres, amp[:, h, c])
            p = np.interp(n, centres, phase[:, h, c])
            out[:, c] += a * np.cos(p)
    return out


# ------------------------------------------------------------------------------- the bridge

def bridge_attack(x: np.ndarray, fs: int, J: int, loop: np.ndarray, f0: list | float, *,
                  bridge_s: float = 0.3, hop: int | None = None, nh: int | None = None,
                  max_hz: float = 12000.0, periods: float = 6.0, amp_floor_db: float = -100.0) -> tuple[np.ndarray, dict]:
    """Copy of ``x`` whose [J - B, J) region has every harmonic band relaxed onto the loop's own
    trajectory for that band, so that at J (where the loop's sample 0 will sit) amplitude, phase
    and frequency are exactly the loop's.  Samples before J - B are untouched.  Returns
    (x_bridged, diagnostics)."""
    if x.ndim == 1:
        x = x[:, None]
    if loop.ndim == 1:
        loop = loop[:, None]
    N, C = x.shape
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    f0m = float(np.exp(np.mean(np.log(f0s))))
    if nh is None:
        nh = int(min(64, max(1, math.floor(min(max_hz, 0.45 * fs) / f0m))))
    if hop is None:
        hop = int(max(32, round(fs / f0m)))                          # about one period
    B = int(round(bridge_s * fs))
    M = max(4, B // hop)
    hop = B // M                                                     # frames land exactly on J
    t0 = J - M * hop
    if t0 < 0:
        raise ValueError('bridge starts before the file')
    centres = t0 + np.arange(M + 1) * hop                            # last centre == J

    trk = harmonic_tracks(x, fs, f0s, nh, centres, periods=periods)          # (K, nh, C)
    T, used = loop_band_targets(loop, fs, f0s, nh, centres - J)               # (K, nh, C) complex
    s = raised_cosine(M + 1)[:, None, None]                                   # 0 at t0 -> 1 at J
    floor = 10 ** (amp_floor_db / 20)

    A_rec = np.maximum(trk['amp'], floor)
    A_tgt = np.maximum(np.abs(T), floor)
    amp_m = np.exp((1 - s) * np.log(A_rec) + s * np.log(A_tgt))

    # target total phase, unwrapped with the same convention as the recording's tracks
    p_rec = trk['phase']
    p_tgt = np.zeros_like(p_rec)
    for c in range(C):
        fc = f0s[min(c, len(f0s) - 1)]
        for h in range(1, nh + 1):
            fh = h * fc
            b = T[:, h - 1, c] * np.exp(-2j * np.pi * fh * ((centres - J) / fs))
            p_tgt[:, h - 1, c] = np.unwrap(np.angle(b)) + 2 * np.pi * fh * ((centres - J) / fs)
    D = p_tgt - p_rec
    D = D - 2 * np.pi * np.round(D[-1:] / (2 * np.pi))                        # |D(J)| <= pi
    p_m = p_rec + s * D

    y_u = _synth(trk['amp'], p_rec, centres, t0, J)
    y_m = _synth(amp_m, p_m, centres, t0, J)
    out = x.copy()
    out[t0:J] += y_m - y_u

    step_db = 20 * np.log10(A_tgt[-1] / A_rec[-1])
    weight = A_tgt[-1] / (A_tgt[-1].sum(axis=0, keepdims=True) + 1e-20)
    info = dict(bridge_start=int(t0), bridge_samples=int(J - t0), frames=int(M + 1), hop=int(hop), nh=int(nh),
                bins_per_band_max=int(max((len(u) for u in used), default=0)),
                amp_move_db_weighted=float(np.sum(np.abs(step_db) * weight) / C),
                amp_move_db_max=float(np.max(np.abs(step_db[:12]))) if nh >= 12 else float(np.max(np.abs(step_db))),
                phase_move_deg_mean=float(np.mean(np.abs(np.degrees(D[-1])))),
                model_fit_db=float(20 * np.log10(_rms(x[t0:J] - y_u) / _rms(x[t0:J]))),
                correction_rms_db=float(20 * np.log10(_rms(y_m - y_u) / _rms(x[t0:J]))))
    return out, info


# ------------------------------------------------------------------------------- where to join

def find_join_bridge(x: np.ndarray, fs: int, loop: np.ndarray, f0: list | float, lo: int, hi: int, *,
                     nh: int = 16, step_s: float = 0.005, time_weight_db_per_s: float = 2.0,
                     periods: float = 6.0) -> dict:
    """With the bridge doing the phase work, the best join is where the recording's harmonic
    amplitudes are already closest to the loop's (least to morph), with a mild preference for
    earlier joins.  cost(J) = loop-amplitude-weighted mean |dB difference| + time_weight * (J - lo)."""
    if loop.ndim == 1:
        loop = loop[:, None]
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    C = loop.shape[1]
    centres = np.arange(lo, hi + 1, max(1, int(step_s * fs)))
    T, _ = loop_band_targets(loop, fs, f0s, nh, np.zeros(1))
    A_tgt = np.abs(T[0])                                                       # (nh, C) at loop sample 0
    trk = harmonic_tracks(x, fs, f0s, nh, centres, periods=periods)
    floor = 1e-5
    d = np.abs(20 * np.log10(np.maximum(trk['amp'], floor) / np.maximum(A_tgt, floor)[None]))   # (K, nh, C)
    w = (A_tgt / (A_tgt.sum(axis=0, keepdims=True) + 1e-20))[None]
    cost_amp = np.sum(d * w, axis=(1, 2)) / C
    cost = cost_amp + time_weight_db_per_s * (centres - lo) / fs
    k = int(np.argmin(cost))
    return dict(J=int(centres[k]), amp_distance_db=float(cost_amp[k]), cost=float(cost[k]),
                best_amp_distance_db=float(cost_amp.min()), n_candidates=int(len(centres)))


# ------------------------------------------------------------------------------- did it work?

def continuity_metrics(out: np.ndarray, x: np.ndarray, J: int, loop_start: int, fs: int, f0: list | float,
                       nh: int = 12, periods: float = 8.0, band_ms: float = 100.0,
                       loop: np.ndarray | None = None) -> dict:
    """Per-harmonic amplitude step (dB) and phase error (deg) across the join, over ``periods``
    periods each side, and third-octave band steps over ``band_ms`` each side — for the assembled
    file at loop_start, for the recording at J and, if ``loop`` is given, for the loop tiled
    against itself at its own seam (``loop_*``: a perfectly continuous reference).

    Read the harmonic numbers as *excess over the loop's own*: a band that holds several grid
    partials (vibrato, a section's chorus) has a combined phase that wanders within a few periods
    even in a perfectly continuous signal, so the absolute figures are not zero there.
    ``excess_harm_step_db`` and ``excess_phase_err_deg`` are the join minus the loop's own."""
    from scipy.signal import welch
    if out.ndim == 1:
        out = out[:, None]
    if x.ndim == 1:
        x = x[:, None]
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    f0m = float(np.exp(np.mean(np.log(f0s))))
    n = int(round(periods * fs / f0m))
    C = out.shape[1]

    def harm(seg, c):
        Wn = len(seg)
        w = get_window('hann', Wn, fftbins=True)
        fc = f0s[min(c, len(f0s) - 1)]
        t = np.arange(Wn) / fs
        return np.array([np.sum(seg[:, c] * w * np.exp(-2j * np.pi * h * fc * t)) * 2 / w.sum() for h in range(1, nh + 1)])

    def steps(sig, p):
        a_steps, ph_errs = [], []
        for c in range(C):
            before = harm(sig[p - n: p], c)
            after = harm(sig[p: p + n], c)
            fc = f0s[min(c, len(f0s) - 1)]
            adv = np.exp(2j * np.pi * np.arange(1, nh + 1) * fc * n / fs)
            a_steps.append(20 * np.log10((np.abs(after) + 1e-9) / (np.abs(before) + 1e-9)))
            ph_errs.append(np.degrees(np.abs(np.angle(after * np.conj(before * adv)))))
        return np.array(a_steps), np.array(ph_errs)

    a_out, p_out = steps(out, loop_start)
    a_rec, p_rec = steps(x, J)

    def bands(sig, p):
        W = int(band_ms * 1e-3 * fs)
        def b(seg):
            f, P = welch(seg.mean(1), fs, nperseg=min(2048, len(seg)))
            lo, hi = 100.0, min(12000.0, 0.45 * fs)
            edges = lo * 2.0 ** (np.arange(0, int(3 * math.log2(hi / lo)) + 2) / 3.0)
            return np.array([10 * np.log10(P[(f >= a) & (f < bb)].sum() + 1e-20) for a, bb in zip(edges[:-1], edges[1:])
                             if ((f >= a) & (f < bb)).sum()])
        return b(sig[p: p + W]) - b(sig[p - W: p])

    b_out, b_rec = bands(out, loop_start), bands(x, J)
    wgt = np.abs(np.array([harm(out[loop_start: loop_start + n], c) for c in range(C)]))
    wgt = wgt / (wgt.sum() + 1e-20)
    res = dict(harm_step_db_max=float(np.max(np.abs(a_out))), harm_step_db_wmean=float(np.sum(np.abs(a_out) * wgt)),
               harm_phase_err_deg_wmean=float(np.sum(p_out * wgt)), harm_phase_err_deg_max=float(np.max(p_out)),
               rec_harm_step_db_max=float(np.max(np.abs(a_rec))), rec_harm_step_db_wmean=float(np.sum(np.abs(a_rec) * wgt)),
               rec_harm_phase_err_deg_wmean=float(np.sum(p_rec * wgt)),
               band_step_db_mean=float(np.mean(np.abs(b_out))), band_step_db_max=float(np.max(np.abs(b_out))),
               rec_band_step_db_mean=float(np.mean(np.abs(b_rec))), rec_band_step_db_max=float(np.max(np.abs(b_rec))))
    if loop is not None:
        if loop.ndim == 1:
            loop = loop[:, None]
        L = len(loop)
        tiled = np.tile(loop, (3, 1))
        a_lp, p_lp = steps(tiled, L)
        b_lp = bands(tiled, L) if L >= int(band_ms * 1e-3 * fs) else np.zeros(1)
        res.update(loop_harm_step_db_wmean=float(np.sum(np.abs(a_lp) * wgt)), loop_harm_phase_err_deg_wmean=float(np.sum(p_lp * wgt)),
                   loop_band_step_db_mean=float(np.mean(np.abs(b_lp))),
                   excess_harm_step_db=float(res['harm_step_db_wmean'] - np.sum(np.abs(a_lp) * wgt)),
                   excess_phase_err_deg=float(res['harm_phase_err_deg_wmean'] - np.sum(p_lp * wgt)))
    # sample-domain check: does the last `periods` periods of the file match what precedes loop[0]?
    if loop is not None and loop_start >= n:
        tail, entry = out[loop_start - n: loop_start], np.tile(loop, (2, 1))[len(loop) - n: len(loop)]
        num = sum(np.dot(tail[:, c], entry[:, c]) for c in range(C))
        den = math.sqrt(sum(np.dot(tail[:, c], tail[:, c]) for c in range(C)) * sum(np.dot(entry[:, c], entry[:, c]) for c in range(C))) + 1e-20
        res['tail_ncc'] = float(num / den)
        res['tail_residual_db'] = float(20 * np.log10(_rms(tail - entry) / _rms(entry)))
    return res
