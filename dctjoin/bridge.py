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
* ``bridge_attack``       write each band as envelope x carrier: the recording's as z_rec(t)
                          exp(i h phi1(t)) with phi1 the tracked fundamental phase, the loop's
                          as z_tgt(t) exp(i psi(t)) with psi the phase of the band's content
                          smoothed over 25 ms (so a frozen vibrato pattern is in the carrier,
                          not spinning the envelope).  Both carriers are smooth by
                          construction, so their difference G is smooth.  Then,
                          with s(t) a raised cosine from 0 at J-B to 1 at J:

                              carrier_m = h phi1 + s * G,           |G(J)| <= pi
                              env_m     = (1-s)^p z_rec e^{i s d(t)} + s^p z_tgt

                          d(t) is the phase difference between the two envelopes, from a
                          smoothed amplitude-weighted cross product (nulls carry no weight, so
                          they cannot flip it); rotating the recording's envelope by s d(t)
                          keeps the pair within a quarter turn while both are audible, and p,
                          between 1 and 1/2 from |d|, makes a coherent pair cross linearly and
                          a quadrature pair at equal power: no dip, no bump mid-bridge.  The blend
                          is done in the complex plane, so a band that passes through an
                          amplitude null (a section's chorus does, constantly) is a smooth curve
                          rather than a phase jump — the failure mode of any polar morph.
                          The band is synthesised twice, measured and morphed, and the
                          difference is added to the recording: zero at J-B, the loop's exact
                          amplitude, phase and frequency at J, and everything the model does
                          not describe — noise, the transient, fluctuations faster than the
                          analysis window — passes through unchanged.  Vibrato and tremolo die
                          away into the loop's own motion instead of stopping dead.
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
    used per (h, c) and fpk[h-1, c], the frequency of the band's strongest partial."""
    if loop.ndim == 1:
        loop = loop[:, None]
    L, C = loop.shape
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    Y = np.fft.rfft(loop, axis=0)
    amp = 2.0 * np.abs(Y) / L
    ph = np.angle(Y)
    fbin = np.arange(len(Y)) * fs / L
    T = np.zeros((len(n_rel), nh, C), dtype=complex)
    fpk = np.zeros((nh, C))
    used = []
    for c in range(C):
        fc = f0s[min(c, len(f0s) - 1)]
        for h in range(1, nh + 1):
            sel = np.where((fbin >= (h - 0.5) * fc) & (fbin < (h + 0.5) * fc) & (np.arange(len(Y)) >= 1))[0]
            if not sel.size:
                used.append([])
                fpk[h - 1, c] = h * fc
                continue
            a = amp[sel, c]
            keep = sel[a >= a.max() * 10 ** (rel_db / 20)]
            used.append([int(m) for m in keep])
            fpk[h - 1, c] = fbin[sel[int(np.argmax(a))]]
            # sum of partials  a_m exp(i (2 pi m n / L + phi_m))  at every n in n_rel
            T[:, h - 1, c] = np.sum(amp[keep, c][None, :] * np.exp(1j * (2 * np.pi * np.outer(n_rel, keep) / L + ph[keep, c][None, :])), axis=1)
    return T, used, fpk


# ------------------------------------------------------------------------------- the recording's harmonics

def _f0_track(x: np.ndarray, fs: int, fc: float, c: int, centres: np.ndarray, periods: float,
              nh_pitch: int = 4, max_dev: float = 0.1) -> np.ndarray:
    """Per-frame fundamental frequency of channel ``c`` from the first ``nh_pitch`` harmonics
    (fixed carriers, baseband phase increments, amplitude-squared weighted, 3-frame smoothed),
    clipped to +-``max_dev`` of the nominal ``fc``."""
    N = x.shape[0]
    K = len(centres)
    if K < 2:
        return np.full(K, fc)
    W = int(round(periods * fs / fc)) | 1
    half = W // 2
    w = get_window('hann', W, fftbins=False)
    n_rel = np.arange(-half, half + 1)
    idx = centres[:, None] + n_rel[None, :]
    valid = (idx >= 0) & (idx < N)
    frames = np.where(valid, x[np.clip(idx, 0, N - 1), c], 0.0) * w[None, :]
    dt = np.diff(centres) / fs
    num = np.zeros(K - 1)
    den = np.zeros(K - 1)
    for h in range(1, nh_pitch + 1):
        fh = h * fc
        z = np.sum(frames * np.exp(-2j * np.pi * fh * (n_rel / fs))[None, :], axis=1)
        z = z * np.exp(-2j * np.pi * fh * (centres / fs))
        dph = np.angle(z[1:] * np.conj(z[:-1]))
        f_int = (fh + dph / (2 * np.pi * dt)) / h                       # implied fundamental per interval
        wt = (np.abs(z[1:]) * np.abs(z[:-1])) * h                        # strong, high harmonics resolve pitch best
        num += wt * f_int
        den += wt
    f_int = np.where(den > 0, num / np.maximum(den, 1e-30), fc)
    f_int = np.clip(f_int, fc * (1 - max_dev), fc * (1 + max_dev))
    f = np.concatenate([[f_int[0]], 0.5 * (f_int[1:] + f_int[:-1]), [f_int[-1]]])
    # smooth over a fixed time (25 ms), not a fixed number of frames: at a high f0 the hop is under
    # a millisecond and a 3-frame average leaves jitter that the morph would print onto the loop's
    # component as sidebands; 25 ms still passes vibrato untouched
    hop_s = float(np.median(np.diff(centres))) / fs if K > 1 else 0.005
    kk = int(max(3, round(0.025 / hop_s)) | 1)
    if K >= 3:
        kern = get_window('hann', kk, fftbins=False)
        kern = kern / kern.sum()
        f = np.convolve(np.pad(f, kk // 2, mode='edge'), kern, mode='valid')[:K]
    return f


def harmonic_tracks(x: np.ndarray, fs: int, f0: list | float, nh: int, centres: np.ndarray,
                    periods: float = 6.0) -> dict:
    """Pitch-following heterodyne analysis of ``x`` (N, C) at frame ``centres`` (samples).

    Pass 1 tracks the fundamental f0(t) from the first harmonics; pass 2 demodulates harmonic
    h with the carrier h * phi1(t), phi1 the fundamental's accumulated phase, so a harmonic
    stays inside its analysis band however far the pitch wobbles (harmonic 100 of a 41 Hz
    note moves 20 Hz for a 5-cent wobble; a fixed carrier would lose it).  Per band h and
    channel c: amp[k, h, c], unwrapped total phase phase[k, h, c] (continuous across frames)
    and instantaneous frequency freq[k, h, c]; also f0_track[k, c], the complex baseband envelope
    bb[k, h, c] and the analytic carrier phase carrier[k, h, c] = h * phi1(c_k) (bb * exp(i carrier)
    is the band's analytic signal at the frame centre; nothing here needs unwrapping)."""
    if x.ndim == 1:
        x = x[:, None]
    N, C = x.shape
    f0s = np.atleast_1d(np.asarray(f0, dtype=float))
    K = len(centres)
    amp = np.zeros((K, nh, C))
    phase = np.zeros((K, nh, C))
    freq = np.zeros((K, nh, C))
    bb = np.zeros((K, nh, C), dtype=complex)
    carrier = np.zeros((K, nh, C))
    f0_track = np.zeros((K, C))
    for c in range(C):
        fc = f0s[min(c, len(f0s) - 1)]
        W = int(round(periods * fs / fc)) | 1
        half = W // 2
        w = get_window('hann', W, fftbins=False)
        wsum = w.sum()
        n_rel = np.arange(-half, half + 1)
        idx = centres[:, None] + n_rel[None, :]
        valid = (idx >= 0) & (idx < N)
        frames = np.where(valid, x[np.clip(idx, 0, N - 1), c], 0.0) * w[None, :]
        # fundamental phase per sample over the span the frames cover
        f0t = _f0_track(x, fs, fc, c, centres, periods)
        f0_track[:, c] = f0t
        n0, n1 = int(idx.min()), int(idx.max()) + 1
        n_all = np.arange(n0, n1)
        f_all = np.interp(n_all, centres, f0t)
        phi1 = 2 * np.pi * np.concatenate([[0.0], np.cumsum(f_all)[:-1]]) / fs      # phase at n = sum_{i<n} f_i
        phi_frames = phi1[idx - n0]                                                # (K, W)
        phi_c = phi1[centres - n0]                                                 # (K,)
        E1 = np.exp(-1j * (phi_frames - phi_c[:, None]))                           # carrier for h = 1, per frame
        Eh = np.ones_like(E1)
        for h in range(1, nh + 1):
            Eh = Eh * E1                                                           # carrier for h
            z = np.sum(frames * Eh, axis=1) * (2.0 / wsum)                         # A exp(i(theta + h phi1(c_k)))
            zb = z * np.exp(-1j * h * phi_c)                                       # baseband: A exp(i theta)
            bb[:, h - 1, c] = zb
            carrier[:, h - 1, c] = h * phi_c
            amp[:, h - 1, c] = np.abs(zb)
            phase[:, h - 1, c] = np.unwrap(np.angle(zb)) + h * phi_c
            if K > 1:
                dph = np.angle(zb[1:] * np.conj(zb[:-1]))
                dt = np.diff(centres) / fs
                f_int = h * 0.5 * (f0t[1:] + f0t[:-1]) + dph / (2 * np.pi * dt)
                freq[:, h - 1, c] = np.concatenate([[f_int[0]], 0.5 * (f_int[1:] + f_int[:-1]), [f_int[-1]]])
            else:
                freq[:, h - 1, c] = h * fc
    return dict(amp=amp, phase=phase, freq=freq, centres=centres, f0_track=f0_track, bb=bb, carrier=carrier)


def _synth(env: np.ndarray, carrier: np.ndarray, centres: np.ndarray, n0: int, n1: int) -> np.ndarray:
    """Sum of bands over samples [n0, n1) from a frame-rate complex envelope env (K, nh, C) and an
    analytic carrier phase carrier (K, nh, C): Re(env * exp(i carrier)).  The envelope is
    interpolated linearly in the complex plane (smooth through nulls, no phase unwrapping), the
    carrier linearly (exact for a steady partial when the hop is about a period or less)."""
    n = np.arange(n0, n1)
    K, nh, C = env.shape
    out = np.zeros((n1 - n0, C))
    for c in range(C):
        for h in range(nh):
            re = np.interp(n, centres, env[:, h, c].real)
            im = np.interp(n, centres, env[:, h, c].imag)
            p = np.interp(n, centres, carrier[:, h, c])
            out[:, c] += re * np.cos(p) - im * np.sin(p)
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
        nh = int(min(200, max(1, math.floor(min(max_hz, 0.45 * fs) / f0m))))
    if hop is None:
        # one period, but never more than 5 ms: at low f0 a harmonic band spans many loop-grid
        # bins and its combined trajectory moves faster than once per period
        hop = int(max(32, min(round(fs / f0m), round(0.005 * fs))))
    B = int(round(bridge_s * fs))
    M = max(4, B // hop)
    hop = B // M                                                     # frames land exactly on J
    t0 = J - M * hop
    if t0 < 0:
        raise ValueError('bridge starts before the file')
    centres = t0 + np.arange(M + 1) * hop                            # last centre == J

    trk = harmonic_tracks(x, fs, f0s, nh, centres, periods=periods)          # (K, nh, C)
    T, used, fpk = loop_band_targets(loop, fs, f0s, nh, centres - J)          # (K, nh, C) complex
    s = raised_cosine(M + 1)[:, None, None]                                   # 0 at t0 -> 1 at J
    floor = 10 ** (amp_floor_db / 20)

    # carriers.  The recording's is h * phi1(t) from the smoothed pitch track, so its vibrato is in
    # the carrier and its envelope is slow.  The loop's band gets the same treatment: the phase of
    # its content smoothed over 25 ms (a frozen vibrato pattern spins the band's phase at tens of
    # hertz; left in the envelope it would beat against the recording's still envelope mid-bridge)
    ph_rec = trk['carrier']
    ph_nom = 2 * np.pi * fpk[None] * ((centres - J) / fs)[:, None, None]      # nominal: strongest partial
    base = T * np.exp(-1j * ph_nom)                                            # slow baseband of the band
    kc = int(max(3, round(0.025 * fs / hop)) | 1)
    kern_c = get_window('hann', kc, fftbins=False)
    kern_c = kern_c / kern_c.sum()
    bpad = np.concatenate([np.repeat(base[:1], kc // 2, axis=0), base, np.repeat(base[-1:], kc // 2, axis=0)], axis=0)
    bs = np.zeros_like(base)
    for i in range(kc):
        bs += kern_c[i] * bpad[i: i + len(base)]
    ph_tgt = ph_nom + np.unwrap(np.angle(bs), axis=0)                          # carrier = nominal + smoothed deviation
    G = ph_tgt - ph_rec
    G = G - 2 * np.pi * np.round(G[-1:] / (2 * np.pi))                        # |G(J)| <= pi
    ph_m = ph_rec + s * G                                                     # carrier glides onto the loop's

    # envelopes relative to those carriers.  The recording's is rotated progressively by the
    # residual phase difference at J, so the dominant component arrives in phase; the blend is
    # done in the complex plane (a band passing through a null is a smooth curve there, not a
    # phase jump), with a crossfade law between linear (coherent) and equal-power (quadrature)
    # chosen from that residual difference so the magnitude neither dips nor bumps mid-bridge
    zb = trk['bb']
    bt = T * np.exp(-1j * ph_tgt)
    # phase difference between the envelopes, per frame, from a smoothed amplitude-weighted cross
    # product (a null in either envelope carries no weight, so it cannot flip the estimate); the
    # smoothed sequence changes slowly and unwraps safely.  The recording's envelope is rotated by
    # s * d(t), so while both are audible the pair stays within a quarter turn: no cancellation
    cross = bt * np.conj(zb)
    kk = max(3, int(round(0.05 * fs / hop)) | 1)
    kern = get_window('hann', kk, fftbins=False)
    kern = kern / kern.sum()
    cpad = np.concatenate([np.repeat(cross[:1], kk // 2, axis=0), cross, np.repeat(cross[-1:], kk // 2, axis=0)], axis=0)
    cs = np.zeros_like(cross)
    for i in range(kk):
        cs += kern[i] * cpad[i: i + len(cross)]
    d = np.unwrap(np.angle(cs), axis=0)
    d_end = ((d[-1:] + np.pi) % (2 * np.pi)) - np.pi                          # (1, nh, C), for the diagnostics
    zr = zb * np.exp(1j * s * d)
    dw = np.abs(((d + np.pi) % (2 * np.pi)) - np.pi)
    pw = 1.0 - dw / (2 * np.pi)                                                # 1 -> linear, 0.5 -> equal power
    w_r = (1 - s) ** pw
    w_t = s ** pw
    env_m = w_r * zr + w_t * bt

    A_rec = np.maximum(trk['amp'], floor)
    A_tgt = np.maximum(np.abs(T), floor)
    y_u = _synth(zb, ph_rec, centres, t0, J)
    y_m = _synth(env_m, ph_m, centres, t0, J)
    out = x.copy()
    out[t0:J] += y_m - y_u
    D = d_end[0]
    # interior glitch check: the largest spectral-flux event inside the bridge, relative to the
    # recording's own largest over the same span, worst channel (> ~3 dB would mean the morph
    # added an event).  Per channel, not the mono sum: the loop's inter-channel phases differ
    # from the recording's, so the bridge glides each harmonic's L-R phase, which changes the
    # mono sum smoothly but is no event in either channel
    # The reference is the larger of the recording's own largest event and the loop's own (the
    # loop tiled over the same span): the bridge fades the loop's texture in, so a loop that
    # itself beats — vibrato sidebands outside dctloop's lock become separate partials — shows
    # its own modulation inside the bridge, which is the loop's character, not a bridge defect.
    # loop_flux_vs_rec_db says how much more eventful the loop is than the recording.
    from .join import _flux
    tiled = np.tile(loop, (int(np.ceil((J - t0) / len(loop))) + 1, 1))[: J - t0]
    vals, lv = [], []
    for c in range(C):
        fo, _ = _flux(out[t0:J, c], fs)
        fx, _ = _flux(x[t0:J, c], fs)
        fl, _ = _flux(tiled[:, c], fs)
        if fo.size and fx.size and fl.size:
            vals.append(20 * np.log10((fo.max() + 1e-9) / (max(fx.max(), fl.max()) + 1e-9)))
            lv.append(20 * np.log10((fl.max() + 1e-9) / (fx.max() + 1e-9)))
    bridge_flux_db = float(max(vals)) if vals else 0.0
    loop_flux_vs_rec_db = float(max(lv)) if lv else 0.0

    step_db = 20 * np.log10(A_tgt[-1] / A_rec[-1])
    weight = A_tgt[-1] / (A_tgt[-1].sum(axis=0, keepdims=True) + 1e-20)
    info = dict(bridge_start=int(t0), bridge_samples=int(J - t0), frames=int(M + 1), hop=int(hop), nh=int(nh),
                bins_per_band_max=int(max((len(u) for u in used), default=0)),
                amp_move_db_weighted=float(np.sum(np.abs(step_db) * weight) / C),
                amp_move_db_max=float(np.max(np.abs(step_db[:12]))) if nh >= 12 else float(np.max(np.abs(step_db))),
                phase_move_deg_mean=float(np.mean(np.abs(np.degrees(D)))), bridge_flux_db=bridge_flux_db,
                loop_flux_vs_rec_db=loop_flux_vs_rec_db,
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
    T, _, _ = loop_band_targets(loop, fs, f0s, nh, np.zeros(1))
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
        if Wn < 2:                      # no window fits: report nothing rather than raising
            return np.zeros(nh, dtype=complex)
        w = get_window('hann', Wn, fftbins=True)
        fc = f0s[min(c, len(f0s) - 1)]
        t = np.arange(Wn) / fs
        return np.array([np.sum(seg[:, c] * w * np.exp(-2j * np.pi * h * fc * t)) * 2 / w.sum() for h in range(1, nh + 1)])

    def steps(sig, p):
        a_steps, ph_errs = [], []
        for c in range(C):
            before = harm(sig[max(0, p - n): p], c)      # clamp: a low note's window can be longer than what precedes p
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
        # enough copies that a full n-sample window fits on both sides of the seam being measured:
        # with a short loop and a low fundamental, n can be several times L
        reps = max(3, 2 * int(math.ceil(n / L)) + 1)
        tiled = np.tile(loop, (reps, 1))
        seam = L * (reps // 2)
        a_lp, p_lp = steps(tiled, seam)
        b_lp = bands(tiled, seam) if L >= int(band_ms * 1e-3 * fs) else np.zeros(1)
        res.update(loop_harm_step_db_wmean=float(np.sum(np.abs(a_lp) * wgt)), loop_harm_phase_err_deg_wmean=float(np.sum(p_lp * wgt)),
                   loop_band_step_db_mean=float(np.mean(np.abs(b_lp))),
                   excess_harm_step_db=float(res['harm_step_db_wmean'] - np.sum(np.abs(a_lp) * wgt)),
                   excess_phase_err_deg=float(res['harm_phase_err_deg_wmean'] - np.sum(p_lp * wgt)))
    # sample-domain check: does the file just before the join match what precedes loop[0]?  A fixed
    # short span (at most 25 ms): on a low note 8 periods would cover most of the bridge, i.e. the
    # deliberate blend, not the landing
    nt = int(min(n, round(0.025 * fs)))
    if loop is not None and loop_start >= nt:
        tail, entry = out[loop_start - nt: loop_start], np.tile(loop, (2, 1))[len(loop) - nt: len(loop)]
        num = sum(np.dot(tail[:, c], entry[:, c]) for c in range(C))
        den = math.sqrt(sum(np.dot(tail[:, c], tail[:, c]) for c in range(C)) * sum(np.dot(entry[:, c], entry[:, c]) for c in range(C))) + 1e-20
        res['tail_ncc'] = float(num / den)
        res['tail_residual_db'] = float(20 * np.log10(_rms(tail - entry) / _rms(entry)))
    return res
