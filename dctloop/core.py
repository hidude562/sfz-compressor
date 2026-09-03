"""dctloop.core — the loop itself: analysis on the loop grid, DCT or DFT synthesis.

The idea
--------
A loop of L samples can only contain sinusoids whose wavelength divides L, i.e. the
frequencies m * fs / L ("the loop grid").  Rebuild the sound from those alone and it is
L-periodic by construction: no seam, no crossfade.  The cosine of grid line m is

    cos(pi * m * (2n + 1) / L)          (wavelength L / m samples, m whole cycles per loop)

which is exactly basis function m of an inverse DCT-II of L/2 points.  Every such cosine is
even about n = -1/2 and about n = L/2 - 1/2, so a DCT loop is a palindrome: synthesise L/2
samples and append their reversal.  The DFT basis uses the same grid with complex phases and no
mirror.

Analysis (``analyse_on_grid``)
------------------------------
A real cosine transform cannot measure a partial's amplitude on its own — one DCT bin holds
only the cosine projection, the sine part leaks into the neighbours — so the analysis uses the
DCT together with its quadrature twin, the DST (together: the DFT), and only the synthesis is a
pure cosine transform.  It is a Welch periodogram on frames of exactly 2L samples with a
periodic Hann window, whose spectrum is zero at every multiple of 1/L: an on-grid partial lands
in exactly one grid bin whatever its phase.  The averaged periodogram is then distributed over
the grid (``_assign_to_grid``):

1. harmonics are *locked*: everything within +-lock_width grid bins of h * f0 (plus the
   falling skirt) goes to bin round(h * f0 * L / fs);
2. every other spectral peak is moved *whole* to its nearest grid bin (never split between two
   bins, which would beat at 1/L Hz);
3. the remaining noise floor is shared between neighbouring bins in proportion to their energy.

Every periodogram bin is used exactly once, so energy is conserved: the loop has the input's
power spectrum on the grid (white noise comes back at unity gain, a harmonic of amplitude 1 as
1.000).

Loop length (``fit_loop_length``)
---------------------------------
L is rounded so that an integer number K of fundamental periods fits as exactly as possible;
harmonic h then sits on grid bin h * K.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import idct, rfft, irfft
from scipy.signal.windows import get_window

BASES = ('dct', 'dft')


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-20)


def fit_loop_length(seconds: float, fs: int, f0: float, search: int = 2) -> tuple[int, int, float]:
    """Even loop length L (samples) near ``seconds`` that holds an integer number K of periods
    of f0 as exactly as possible.  Returns (L, K, pitch error of the grid in cents)."""
    P = fs / f0
    K0 = max(1, int(round(seconds * fs / P)))
    best = None
    for K in range(max(1, K0 - search), K0 + search + 1):
        L = 2 * int(round(K * P / 2))
        if L < 2:
            continue
        cents = 1200 * math.log2((K * fs / L) / f0)
        cand = (abs(cents), abs(K - K0), L, K, cents)
        if best is None or cand < best:
            best = cand
    return best[2], best[3], best[4]


def harmonic_bins(f0: float, L: int, fs: int) -> np.ndarray:
    """Grid bins round(h * f0 * L / fs) of all harmonics below Nyquist."""
    kf = f0 * L / fs
    M = L // 2
    hmax = int(M / kf) if kf > 0 else 0
    return np.round(np.arange(1, hmax + 1) * kf).astype(int)


def _assign_to_grid(P: np.ndarray, lock: np.ndarray | None = None, lock_width: float = 1.5,
                    npeaks_diag: int = 20) -> tuple[np.ndarray, float]:
    """Distribute one channel's averaged 2L-point periodogram P (bins k * fs / (2L), k = 0..L)
    onto the loop grid m = k / 2 — see the module docstring for the three steps.  Returns the
    grid energies and, as a diagnostic, the energy-weighted distance of the strongest peaks
    from the grid (0 = all on grid, 0.25 = random)."""
    nb = len(P)
    M = (nb - 1) // 2
    E = np.zeros(M + 1)
    taken = np.zeros(nb, bool)

    # 1. harmonic locking
    if lock is not None and lock_width > 0:
        w = int(round(2 * lock_width))
        for m in np.asarray(lock, dtype=int):
            if m < 1 or m >= M:
                continue
            lo, hi = max(0, 2 * m - w), min(nb - 1, 2 * m + w)
            while lo > 0 and not taken[lo - 1] and P[lo - 1] < P[lo]:
                lo -= 1
            while hi < nb - 1 and not taken[hi + 1] and P[hi + 1] < P[hi]:
                hi += 1
            sel = np.arange(lo, hi + 1)
            sel = sel[~taken[sel]]
            E[m] += P[sel].sum()
            taken[sel] = True

    # 2. every other local maximum: main lobe (+-1.5 bins around the interpolated peak) goes
    #    whole to the nearest grid bin, strongest peaks first
    interior = np.arange(1, nb - 1)
    pk = interior[(P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])]
    grid_offset = 0.0
    if pk.size:
        lp = np.log(P + 1e-30)
        y0, y1, y2 = lp[pk - 1], lp[pk], lp[pk + 1]
        d = np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30), -0.5, 0.5)
        pos = pk + d
        m_near = np.clip(np.round(pos / 2).astype(int), 0, M)
        lo = np.maximum(np.ceil(pos - 1.5).astype(int), 0)
        hi = np.minimum(np.floor(pos + 1.5).astype(int), nb - 1)
        order = np.argsort(P[pk])[::-1]
        for i in order:
            for k in range(lo[i], hi[i] + 1):
                if not taken[k]:
                    taken[k] = True
                    E[m_near[i]] += P[k]
        top = order[:npeaks_diag]
        wt = P[pk[top]]
        off = np.abs(pos[top] / 2 - np.round(pos[top] / 2))
        grid_offset = float(np.average(off, weights=wt)) if wt.sum() > 0 else 0.0

    # 3. the rest: grid bins keep their own energy, half-bins are shared by neighbour energy
    rest = ~taken
    ev = np.arange(0, nb, 2)
    E[ev[rest[ev]] // 2] += P[ev[rest[ev]]]
    od = np.arange(1, nb - 1, 2)
    od = od[rest[od]]
    if od.size:
        left, right = P[od - 1], P[od + 1]
        frac = left / (left + right + 1e-30)
        np.add.at(E, (od - 1) // 2, frac * P[od])
        np.add.at(E, (od + 1) // 2, (1.0 - frac) * P[od])
    return E, grid_offset


def analyse_on_grid(x: np.ndarray, L: int, lock_bins: list | None = None,
                    lock_width: float = 1.5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Amplitude of the sound at every loop-grid frequency m * fs / L (m = 0..L/2) per channel,
    a complex reference value per bin (the phase / sign used by the synthesis, taken from the
    central frame), and a diagnostics dict.

    x          : (N, C) with N >= 2L
    lock_bins  : per channel, the grid bins of the harmonics (``harmonic_bins``); None = no lock
    """
    if x.ndim == 1:
        x = x[:, None]
    N, C = x.shape
    M = L // 2
    W = 2 * L
    if N < W:
        raise ValueError(f'need at least 2 loop lengths of input (have {N / L:.2f})')
    w = get_window('hann', W, fftbins=True)
    hop = max(1, L // 2)
    starts = np.arange(0, N - W + 1, hop)
    Pbar = np.zeros((L + 1, C))
    ref = None
    for j, s in enumerate(starts):
        X = rfft(x[s:s + W] * w[:, None], axis=0)          # bins k * fs / (2L); the grid is k even
        Pbar += np.abs(X) ** 2
        if j == len(starts) // 2:
            ref = X[0::2][:M + 1]
    Pbar /= len(starts)
    E = np.zeros((M + 1, C))
    offs = []
    for c in range(C):
        lk = None if lock_bins is None else lock_bins[min(c, len(lock_bins) - 1)]
        E[:, c], go = _assign_to_grid(Pbar[:, c], lock=lk, lock_width=lock_width)
        offs.append(go)
    E /= 1.5                                                 # Hann main lobe: 1 + 0.25 + 0.25
    a = 2.0 * np.sqrt(E) / L                                 # |X| = A * sum(w) / 2 = A * L / 2
    return a, ref, dict(frames=int(len(starts)), grid_offset=[round(v, 3) for v in offs])


def synthesise(a: np.ndarray, ref: np.ndarray, L: int, basis: str = 'dct', phase: str = 'orig',
               seed: int = 0) -> np.ndarray:
    """Turn grid amplitudes ``a`` (L/2+1, C) and reference phases ``ref`` into an L-sample loop.

    dct : inverse DCT-II of L/2 points, mirrored (palindrome).  A cosine can only take a sign,
          so the inter-channel phase of every partial is rounded to 0 or pi relative to channel
          0 — keeping whichever relation gives the mono sum closest to the original's, so a
          partial only goes anti-phase when it really was (> ~120 degrees apart).  Choosing each
          channel's sign on its own would scramble the stereo image and cancel partials in mono.
    dft : inverse real FFT with the original (or random) phases, no mirror.
    """
    M = L // 2
    C = a.shape[1]
    rng = np.random.default_rng(seed)
    if basis == 'dct':
        r = ref[:M]
        am = a[:M]
        if phase == 'orig':
            s0 = np.where(r[:, 0].real < 0, -1.0, 1.0)
        elif phase == 'random':
            s0 = rng.choice([-1.0, 1.0], size=M)
        else:
            raise ValueError(phase)
        cosd = (r * np.conj(r[:, :1])).real / (np.abs(r) * np.abs(r[:, :1]) + 1e-30)
        m_orig = np.sqrt(np.maximum(am ** 2 + am[:, :1] ** 2 + 2 * am * am[:, :1] * cosd, 0.0))
        m_same = am + am[:, :1]
        m_opp = np.abs(am - am[:, :1])
        s = s0[:, None] * np.where(np.abs(m_opp - m_orig) < np.abs(m_same - m_orig), -1.0, 1.0)
        s[:, 0] = s0
        # orthonormal IDCT-II of length M: x[n] = sum_m c_m sqrt(2/M) cos(pi m (2n+1) / (2M))
        c = s * am * math.sqrt(M / 2.0)
        half = idct(c, type=2, norm='ortho', n=M, axis=0)
        return np.concatenate([half, half[::-1]], axis=0)
    if basis == 'dft':
        if phase == 'orig':
            ph = np.angle(ref)
        elif phase == 'random':
            ph = rng.uniform(-np.pi, np.pi, (M + 1, C))
        else:
            raise ValueError(phase)
        Y = a * (L / 2.0) * np.exp(1j * ph)                  # irfft amplitude of bin m is 2|Y_m|/L
        return irfft(Y, n=L, axis=0)
    raise ValueError(f'basis must be one of {BASES}, got {basis!r}')


def make_loop(seg: np.ndarray, fs: int, L: int, basis: str = 'dct', phase: str = 'orig',
              lock: float | list | None = None, lock_width: float = 1.5,
              seed: int = 0) -> tuple[np.ndarray, dict]:
    """Build an L-sample loop from ``seg`` (N x C samples of steady sustain, N >= 2L).

    basis      : 'dct' (palindromic loop, real cosines) or 'dft' (complex phases, no mirror)
    phase      : 'orig' takes each partial's phase (dct: sign) from the input, 'random' draws it
    lock       : f0 in Hz (one value or one per channel) for harmonic locking; None disables it
    lock_width : half-width of the lock window in grid bins (default 1.5)
    Returns (loop, info); the loop is scaled to the RMS of the analysed segment, and
    info['gain'] (~1) says how far the calibration was from that.
    """
    if seg.ndim == 1:
        seg = seg[:, None]
    L = int(L)
    if L % 2:
        raise ValueError('loop length must be even')
    q = len(seg) // L
    if q < 2:
        raise ValueError(f'need at least 2 loop lengths of input (have {len(seg) / L:.2f})')
    N = q * L
    off = (len(seg) - N) // 2
    x = seg[off:off + N]
    M = L // 2

    lock_bins = None
    if lock is not None and lock_width > 0:
        lock_bins = [harmonic_bins(float(f), L, fs) for f in np.atleast_1d(np.asarray(lock, dtype=float))
                     if np.isfinite(f) and f > 0]
        lock_bins = lock_bins or None
    a, ref, diag = analyse_on_grid(x, L, lock_bins=lock_bins, lock_width=lock_width)
    a[0] = 0.0                                               # no DC
    a[M] = 0.0                                               # nothing at Nyquist
    loop = synthesise(a, ref, L, basis=basis, phase=phase, seed=seed)
    gain = _rms(x) / _rms(loop)
    loop = loop * gain
    info = dict(N=int(N), q=int(q), L=L, basis=basis, phase=phase, gain=float(gain),
                analysis_offset=int(off), grid_hz=fs / L,
                lock_width=float(lock_width) if lock_bins is not None else 0.0, **diag)
    return loop, info


def loop_signal(x: np.ndarray, fs: int, seconds: float = 1.5, *, basis: str = 'dct',
                f0: float | list | None = None, hint: float | None = None, use_hint: bool = True,
                fit: bool = True, periods: int | None = None, lock: float = 1.5,
                phase: str = 'orig', seed: int = 0) -> tuple[np.ndarray, dict]:
    """Loop an array of steady sustain.  The one-call entry point for signals in memory.

    x        : (N,) or (N, C) float samples of the sustained part of a note (no attack/release)
    seconds  : target loop length; rounded to an integer number of periods (``fit``), and
               shortened if x is shorter than two loops
    basis    : 'dct' or 'dft'
    f0       : fundamental in Hz (one value or one per channel).  None: taken from ``hint``
               (e.g. the note in the file name) when the spectrum bears the hint out, else
               pYIN; ``use_hint=False`` always uses pYIN.  See pitch.f0_per_channel
    periods  : instead of ``seconds``, exactly this many f0 periods
    lock     : harmonic-lock half-width in grid bins (0 disables)
    Returns (loop, info) with info['f0'] per channel, info['K'] periods, info['L'] samples,
    info['grid_cents'] (pitch error of the grid) and the analysis diagnostics.
    """
    from .pitch import f0_per_channel
    if x.ndim == 1:
        x = x[:, None]
    f0c, pitch_info = f0_per_channel(x, fs, f0, hint, use_hint=use_hint, detail=True)
    good = f0c[np.isfinite(f0c)]
    f0_used = float(np.exp(np.mean(np.log(good)))) if good.size else 0.0
    n = len(x)

    if periods and f0_used > 0:
        L, K, cents = fit_loop_length(periods / f0_used, fs, f0_used, search=0)
    elif fit and f0_used > 0:
        L, K, cents = fit_loop_length(seconds, fs, f0_used)
    else:
        L, K, cents = 2 * int(round(seconds * fs / 2)), 0, 0.0
    shortened = False
    if n < 2 * L:
        shortened = True
        T = n / fs / 2 * 0.999
        if fit and f0_used > 0 and not periods:
            L, K, cents = fit_loop_length(T, fs, f0_used)
            while n < 2 * L:
                T *= 0.98
                L, K, cents = fit_loop_length(T, fs, f0_used)
        else:
            L = 2 * int(T * fs / 2)
    lock_f0 = [float(v) if np.isfinite(v) else f0_used for v in f0c] if (f0_used > 0 and lock > 0) else None
    loop, info = make_loop(x, fs, L, basis=basis, phase=phase, lock=lock_f0, lock_width=lock, seed=seed)
    detune = 1200 * math.log2(f0c[-1] / f0c[0]) if (len(f0c) > 1 and good.size == len(f0c)) else 0.0
    info.update(f0=[float(v) for v in f0c], f0_used=f0_used, detune_cents=float(detune), K=int(K),
                grid_cents=float(cents), seconds=L / fs, shortened=shortened, pitch=pitch_info)
    return loop, info
