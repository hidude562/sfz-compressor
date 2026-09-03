"""dctloop.core — seamless loops of a stationary sound via DCT reconstruction.

The idea
--------
Take N = q*L samples of the sustained part of a note (q >= 2, integer), window
them and take the DCT-II.  Bin k of that transform is the cosine

    cos(pi * k * (2n + 1) / (2N)),      wavelength 2N / k samples.

That wavelength divides the loop length L exactly if and only if k is a
multiple of 2N/L = 2q.  So if we rebuild the sound from *only* those bins the
result is exactly L-periodic: it loops by construction, without a crossfade.
Two ways of assigning the retained coefficients are provided:

* ``comb``  keep C[2q*m] as it is and drop everything else (a comb filter on the
            loop grid; in the time domain this is the same as folding the
            windowed signal at period L and averaging, so off-grid content —
            noise, detuned partials — is attenuated by roughly 1/q);
* ``snap``  give loop bin m the *energy* of all DCT bins between its neighbours
            (Parseval: the loop then has the same power spectrum as the input
            on a 1/L Hz grid; noise is kept, "frozen" into an L-periodic
            texture).  The sign comes from the centre bin.

Every retained basis function is an even function about n = -1/2 and about
n = L/2 - 1/2, so a DCT loop is always a palindrome: the second half is the
mirror image of the first.  We therefore synthesise the first L/2 samples with
an inverse DCT-II of length L/2 and append the reversal.  A ``dft`` basis is
offered as well (same grid, complex coefficients, no mirror) for comparison.

The loop length is chosen so that an integer number K of fundamental periods
fits exactly: then every harmonic h*f0 lands on grid bin h*K and nothing is
smeared between two bins (which would beat at 1/L Hz).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf
from scipy.fft import idct, rfft, irfft
from scipy.signal import stft, welch
from scipy.signal.windows import get_window

# --------------------------------------------------------------------------- io / helpers

_NOTE_RE = re.compile(r'(?<![a-z])([a-g])(#|b)?(-?\d)(?![0-9])', re.I)
_SEMITONE = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}


def note_from_name(name: str) -> float | None:
    """Frequency of the last note token ('a#4', 'c3', 'bb2') in a file name, or None."""
    m = None
    for m in _NOTE_RE.finditer(name):
        pass
    if m is None:
        return None
    letter, acc, octave = m.group(1).lower(), m.group(2), int(m.group(3))
    midi = 12 * (octave + 1) + _SEMITONE[letter] + (1 if acc == '#' else -1 if acc == 'b' else 0)
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def load_audio(path: str) -> tuple[np.ndarray, int]:
    x, fs = sf.read(path, dtype='float64', always_2d=True)
    return x, int(fs)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-20)


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


def estimate_f0(seg: np.ndarray, fs: int, hint: float | None = None) -> np.ndarray:
    """Median pYIN f0 per channel (Hz); NaN where unvoiced."""
    import librosa
    if hint:
        fmin, fmax = hint / 1.25, hint * 1.25
    else:
        fmin, fmax = 40.0, 3000.0
    frame = int(2 ** math.ceil(math.log2(4 * fs / fmin)))
    frame = int(min(max(frame, 1024), 16384))
    out = []
    for c in range(seg.shape[1]):
        f0, voiced, _ = librosa.pyin(np.ascontiguousarray(seg[:, c]), fmin=fmin, fmax=fmax, sr=fs,
                                     frame_length=frame, hop_length=frame // 4,
                                     resolution=0.02 if hint else 0.1)
        v = f0[voiced & np.isfinite(f0)]
        out.append(float(np.median(v)) if v.size else float('nan'))
    return np.array(out)


def refine_f0(seg: np.ndarray, fs: int, f0: float, nharm: int = 6, tol: float = 0.03) -> np.ndarray:
    """Per-channel f0 refined from the parabolic-interpolated peaks of harmonics 1..nharm in one
    Hann-windowed spectrum of the whole segment (pYIN is quantised to 10 cents; the integer-period
    loop fit needs ~0.01 cents at a 1.5 s loop)."""
    N = len(seg)
    w = get_window('hann', N, fftbins=True)
    X = np.abs(rfft(seg * w[:, None], axis=0))
    out = []
    for c in range(seg.shape[1]):
        ests, wts = [], []
        for h in range(1, nharm + 1):
            fc = h * f0
            lo, hi = int(fc * (1 - tol) * N / fs), int(fc * (1 + tol) * N / fs) + 1
            if hi >= len(X) - 1 or hi - lo < 3:
                break
            k = lo + int(np.argmax(X[lo:hi, c]))
            if k <= lo or k >= hi - 1:
                continue
            y0, y1, y2 = np.log(X[k - 1:k + 2, c] + 1e-30)
            d = float(np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30), -1, 1))
            ests.append((k + d) * fs / N / h)
            wts.append(float(X[k, c]) ** 2)
        out.append(float(np.average(ests, weights=wts)) if ests else float(f0))
    return np.array(out)


def fit_loop_length(T: float, fs: int, f0: float, search: int = 2) -> tuple[int, int, float]:
    """Even loop length L (samples) near T seconds that holds an integer number K of periods of
    f0 as exactly as possible.  Returns (L, K, pitch error of the grid in cents)."""
    P = fs / f0
    K0 = max(1, int(round(T * fs / P)))
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


# --------------------------------------------------------------------------- the loop itself

def _assign_to_grid(P: np.ndarray, npeaks_diag: int = 20) -> tuple[np.ndarray, float]:
    """Distribute one channel's averaged 2L-point periodogram P (bins k*fs/(2L), k = 0..L) onto
    the loop grid m = k/2.  Every local maximum is a spectral peak: its main lobe (|k - k_peak|
    <= 1.5, k_peak parabolic-interpolated) goes *whole* to the nearest grid bin, so an off-grid
    partial is moved by at most fs/(2L) Hz instead of being split into two components that beat
    at 1/L Hz.  Bins left over (the noise floor between peaks) go to their grid bin if even, or
    are split between the two neighbours in proportion to their energy if odd.  Every bin is
    used exactly once, so energy is conserved.  Also returns the energy-weighted mean distance
    of the strongest peaks from the grid (0 = all on grid, 0.25 = random)."""
    nb = len(P)
    M = (nb - 1) // 2
    E = np.zeros(M + 1)
    taken = np.zeros(nb, bool)
    interior = np.arange(1, nb - 1)
    pk = interior[(P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])]
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
    else:
        grid_offset = 0.0
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


def analyse_on_grid(x: np.ndarray, L: int, mode: str = 'snap') -> tuple[np.ndarray, np.ndarray, dict]:
    """Amplitude of the sound at every loop-grid frequency m*fs/L, m = 0..L/2, per channel,
    plus a complex reference value per bin (for the phase / sign of the synthesis) and a
    diagnostics dict.

    A single real cosine transform cannot measure a partial's amplitude: bin k of the DCT holds
    only the cosine projection, the sine part leaks into the neighbouring bins.  So the analysis
    uses the DCT together with its quadrature twin, the DST — i.e. the complex DFT — and only the
    *synthesis* is a pure cosine transform.

    snap : Welch periodogram on frames of exactly 2L samples (periodic Hann, hop L/2).  That
           window's spectrum is zero at every multiple of 1/L, so an on-grid partial lands in
           exactly one grid bin whatever its phase.  Off-grid peaks are moved whole to their
           nearest grid bin and the noise floor is shared out so that its energy is preserved
           (see _assign_to_grid).
    comb : one Hann frame over the whole N = q*L segment, sampled at bins q*m only.  Also exact
           for on-grid partials, but off-grid content (noise, detuned partials) is attenuated by
           about q/1.5 in power — a gentle de-noising, the time-domain fold-and-average.
    """
    N, C = x.shape
    M = L // 2
    diag: dict = {}
    if mode == 'snap':
        W = 2 * L
        w = get_window('hann', W, fftbins=True)
        hop = max(1, L // 2)
        starts = np.arange(0, N - W + 1, hop)
        Pbar = np.zeros((L + 1, C))
        ref = None
        for j, s in enumerate(starts):
            X = rfft(x[s:s + W] * w[:, None], axis=0)      # bins k*fs/(2L); the grid is k even
            Pbar += np.abs(X) ** 2
            if j == len(starts) // 2:
                ref = X[0::2][:M + 1]
        Pbar /= len(starts)
        E = np.zeros((M + 1, C))
        offs = []
        for c in range(C):
            E[:, c], go = _assign_to_grid(Pbar[:, c])
            offs.append(go)
        E /= 1.5                                             # Hann main lobe: 1 + 0.25 + 0.25
        a = 2.0 * np.sqrt(E) / L                             # |X| = A * sum(w) / 2 = A * L / 2
        diag = dict(frames=int(len(starts)), grid_offset=[round(v, 3) for v in offs])
    elif mode == 'comb':
        q = N // L
        Nq = q * L
        off = (N - Nq) // 2
        w = get_window('hann', Nq, fftbins=True)
        X = rfft(x[off:off + Nq] * w[:, None], axis=0)      # bins k*fs/N; the grid is k = q*m
        ref = X[::q][:M + 1]
        a = 4.0 * np.abs(ref) / Nq                           # |X| = A * sum(w) / 2 = A * N / 4
    else:
        raise ValueError(mode)
    return a, ref, diag


def make_loop(seg: np.ndarray, fs: int, L: int, mode: str = 'snap', basis: str = 'dct',
              phase: str = 'orig', seed: int = 0, window: str | None = None,
              sign: str | None = None) -> tuple[np.ndarray, dict]:
    """Build an L-sample loop from ``seg`` (N x C, N >= 2L).  Returns (loop, info).

    mode  : 'snap' (energy-preserving Welch estimate on the loop grid) or 'comb' (single frame,
            attenuates off-grid content) — see analyse_on_grid
    basis : 'dct' — the loop is an inverse DCT-II of L/2 points mirrored: every basis cosine
            cos(pi*m*(2n+1)/L) has wavelength L/m, so the result is exactly L-periodic (and a
            palindrome).  'dft' — same grid with complex phases, no mirror.
    phase : 'orig' takes each partial's phase (dct: its sign) from the input, 'random' draws it
    """
    if sign is not None:                      # backwards-compatible spelling
        phase = 'orig' if sign in ('center', 'cos', 'orig') else sign
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
    rng = np.random.default_rng(seed)
    M = L // 2

    a, ref, diag = analyse_on_grid(x, L, mode)               # (M+1, C)
    a[0] = 0.0                                               # no DC
    a[M] = 0.0                                               # nothing at Nyquist
    if basis == 'dct':
        # A cosine can only take a sign, so the inter-channel phase of every partial has to be
        # rounded to 0 or pi relative to channel 0.  (Choosing each channel's sign on its own
        # scrambles the stereo image and cancels partials in mono whenever the absolute phase is
        # near 90°.)  The rounding keeps whichever relation gives the mono sum closest to the
        # original's, so a partial only goes anti-phase when it really was (> ~120° apart).
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
        # orthonormal IDCT-II of length M: x[n] = sum_m c_m * sqrt(2/M) * cos(pi*m*(2n+1)/(2M))
        c = s * a[:M] * math.sqrt(M / 2.0)
        half = idct(c, type=2, norm='ortho', n=M, axis=0)
        loop = np.concatenate([half, half[::-1]], axis=0)
    elif basis == 'dft':
        if phase == 'orig':
            ph = np.angle(ref)
        elif phase == 'random':
            ph = rng.uniform(-np.pi, np.pi, ref.shape)
        else:
            raise ValueError(phase)
        Y = a * (L / 2.0) * np.exp(1j * ph)                  # irfft amplitude of bin m is 2|Y_m|/L
        loop = irfft(Y, n=L, axis=0)
    else:
        raise ValueError(basis)

    gain = _rms(x) / _rms(loop)                              # ~1 if the calibration above is right
    loop = loop * gain
    info = dict(N=int(N), q=int(q), L=L, mode=mode, basis=basis, phase=phase, gain=float(gain),
                analysis_offset=int(off), grid_hz=fs / L, **diag)
    return loop, info


# --------------------------------------------------------------------------- metrics

def seam_metrics(loop: np.ndarray, fs: int) -> dict:
    """Spectral-flux outlier test: is the seam (and, for palindromes, the mid-point) any more
    'eventful' than the rest of the loop?  Ratios are max flux at the point / median flux."""
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


def spectrum_match(seg: np.ndarray, fs: int, loop: np.ndarray, lo: float = 60.0,
                   floor_db: float = 60.0) -> dict:
    """Third-octave long-term spectrum of the loop vs the analysed segment, per channel (dB),
    plus the level of the mono sum (which is sensitive to how inter-channel phases were frozen)."""
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
        # bands more than `floor_db` below the loudest band of the reference are inaudible
        # detail (noise floors, dither) and would swamp the average
        d[e1 < e1.max() * 10 ** (-floor_db / 10)] = 0.0
        per_chan.append(d)
    d = np.mean(np.abs(np.array(per_chan)), axis=0)
    mono_db = 10 * np.log10(np.mean(y.mean(axis=1) ** 2) / (np.mean(seg.mean(axis=1) ** 2) + 1e-20) + 1e-20)
    return dict(ltas_mean_abs_db=float(np.mean(d)), ltas_max_abs_db=float(np.max(d)), mono_db=float(mono_db),
                ltas_bands_db=[[round(float(v), 2) for v in ch] for ch in per_chan])


# --------------------------------------------------------------------------- top level

@dataclass
class Result:
    source: str
    fs: int
    f0_hint: float | None
    f0: list
    f0_used: float
    detune_cents: float
    loop_seconds: float
    L: int
    K: int
    grid_cents: float
    seg_start: float
    seg_seconds: float
    info: dict
    seam: dict
    spectrum: dict
    outputs: dict


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


def process(path: str, out_dir: str | None = None, loop: float = 1.5, mode: str = 'snap',
            basis: str = 'dct', phase: str = 'orig', f0: float | None = None,
            fit: bool = True, start: float | None = None, dur: float | None = None,
            periods: int | None = None, preview: bool = True, stem: str | None = None,
            seed: int = 0, verbose: bool = False) -> Result:
    """Load a note, pick its sustain, build the loop, measure it, write the files."""
    import os
    x, fs = load_audio(path)
    name = os.path.splitext(os.path.basename(path))[0]
    stem = stem or name

    # ----- the analysed segment
    if start is not None:
        a = int(start * fs)
        b = int((start + dur) * fs) if dur else len(x)
    else:
        a, b = find_body(x, fs)
        if dur:
            mid = (a + b) // 2
            a, b = max(0, mid - int(dur * fs / 2)), min(len(x), mid + int(dur * fs / 2))
    avail = b - a

    # ----- pitch
    hint = f0 or note_from_name(name)
    f0c = estimate_f0(x[a:b], fs, hint) if f0 is None else np.array([f0] * x.shape[1])
    good = f0c[np.isfinite(f0c)]
    f0_coarse = float(np.exp(np.mean(np.log(good)))) if good.size else (hint or 0.0)
    f0c = refine_f0(x[a:b], fs, f0_coarse) if f0_coarse > 0 else f0c
    good = f0c[np.isfinite(f0c)]
    f0_used = float(np.exp(np.mean(np.log(good)))) if good.size else (hint or 0.0)
    detune = 1200 * math.log2(f0c[-1] / f0c[0]) if (x.shape[1] > 1 and good.size == len(f0c)) else 0.0

    # ----- loop length: an integer number of periods, and at most half the segment
    if periods:
        L, K, cents = fit_loop_length(periods / f0_used, fs, f0_used, search=0)
    elif fit and f0_used > 0:
        L, K, cents = fit_loop_length(loop, fs, f0_used)
    else:
        L, K, cents = 2 * int(round(loop * fs / 2)), 0, 0.0
    if avail < 2 * L:
        T = avail / fs / 2 * 0.999
        if fit and f0_used > 0 and not periods:
            L, K, cents = fit_loop_length(T, fs, f0_used)
            while avail < 2 * L:
                L, K, cents = fit_loop_length(T * 0.98, fs, f0_used)
                T *= 0.98
        else:
            L = 2 * int(T * fs / 2)
        if verbose:
            print(f'  [{name}] only {avail / fs:.2f}s of sustain: loop shortened to {L / fs:.3f}s')
    seg = x[a:b]

    # ----- build + measure
    lp, info = make_loop(seg, fs, L, mode=mode, basis=basis, phase=phase, seed=seed)
    seg_used = seg[info['analysis_offset']:info['analysis_offset'] + info['N']]
    seam = seam_metrics(lp, fs)
    spec = spectrum_match(seg_used, fs, lp)

    # ----- write
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

    res = Result(source=path, fs=fs, f0_hint=hint, f0=[float(v) for v in f0c], f0_used=f0_used,
                 detune_cents=float(detune), loop_seconds=L / fs, L=int(L), K=int(K),
                 grid_cents=float(cents), seg_start=(a + info['analysis_offset']) / fs,
                 seg_seconds=info['N'] / fs, info=info, seam=seam, spectrum=spec, outputs=outputs)
    if out_dir:
        import json
        p_json = os.path.join(out_dir, f'{stem}.json')
        with open(p_json, 'w') as fh:
            json.dump(asdict(res), fh, indent=1)
        outputs['json'] = p_json
    if verbose:
        print(f'  [{name}] f0={f0_used:.2f}Hz (L/R detune {detune:+.1f}c) loop={L / fs:.3f}s '
              f'({K} periods, grid {cents:+.2f}c) seg={info["N"] / fs:.2f}s q={info["q"]} '
              f'seam×{seam["seam_flux_ratio"]:.2f} mid×{seam["mid_flux_ratio"]:.2f} '
              f'p95×{seam["p95_flux_ratio"]:.2f} ltas {spec["ltas_mean_abs_db"]:.2f}dB '
              f'grid-off {info.get("grid_offset")}')
    return res
