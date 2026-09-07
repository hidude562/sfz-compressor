"""dctjoin.join — line a synthesised loop up with the recording at a chosen join and splice.

The problem: the recorded attack must hand over to a loop that dctloop synthesised, without an
audible step.  Three things can step at the join — waveform phase, level, and timbre — and each
gets its own fix:

* ``phase_align_loop``  the loop keeps its grid magnitudes but takes the recording's phase at
                        the join for every grid frequency, so its sample 0 lines up with the
                        recording sample-for-sample (harmonics and moved inharmonic partials
                        alike).  Only a DFT loop can do this; a DCT loop carries signs, not
                        phases, and can only be rotated (``rotate_loop``: circular shift to the
                        best waveform correlation).
* ``level_match``       per-channel RMS over a few periods at the join, not over the following
                        loop length (which is below the join level on anything that decays).
* ``morph_tail``        EQ-morph the last part of the attack towards the loop's spectrum with a
                        raised-cosine ramp, so the timbre arrives at the loop's before the splice.
* ``splice``            recording -> short raised-cosine cross-fade into the loop's last X
                        samples -> loop.  Because the loop is periodic and phase-aligned at its
                        sample 0, its last X samples are what precedes the join, so the fade is
                        between two coherent signals and does not comb-filter.
* ``junction_metrics``  did it work?  Level dip inside the cross-fade and the spectral-flux
                        spike at the join relative to the recording's own.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, stft
from scipy.signal.windows import get_window


def raised_cosine(n: int) -> np.ndarray:
    """0 -> 1 raised-cosine ramp of length n."""
    if n <= 1:
        return np.ones(max(n, 0))
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / (n - 1))


def _rms(a: np.ndarray, axis=None) -> np.ndarray | float:
    return np.sqrt(np.mean(np.square(a), axis=axis)) + 1e-12


# ------------------------------------------------------------------------------- alignment

def phase_align_loop(loop: np.ndarray, x: np.ndarray, join: int, half: int | None = None,
                     min_half: int = 64) -> np.ndarray:
    """Keep the loop's grid magnitudes, take the recording's phases at ``join``.

    The phases come from a *zero-phase* Hann frame centred on the join: ``half`` samples each
    side (default: as many as available up to L), windowed, and circularly shifted so the join
    sample is index 0.  With the window symmetric about index 0 its transform is real and
    positive over the main lobe, so the phase measured at a grid frequency is the phase *at the
    join* of whatever partial is nearest — on the grid or not.  (A causal frame starting at the
    join gets on-grid partials right but skews everything in between by the window's linear
    phase, up to 90 degrees.)  The FFT length is a multiple of L so every grid frequency m*fs/L
    falls exactly on bin m*q, no rounding.
    """
    if loop.ndim == 1:
        loop = loop[:, None]
    if x.ndim == 1:
        x = x[:, None]
    L, C = loop.shape
    h = int(min(L, join, len(x) - join)) if half is None else int(half)
    if h < min_half:
        raise ValueError(f'need at least {min_half} samples each side of the join (have {h})')
    w = get_window('hann', 2 * h, fftbins=True)[:, None]          # symmetric about index h
    fr = x[join - h: join + h] * w
    q = max(2, int(np.ceil(2 * h / L)))
    nfft = q * L
    buf = np.zeros((nfft, C))
    buf[:h] = fr[h:]                                              # the join -> index 0
    buf[nfft - h:] = fr[:h]
    X = np.fft.rfft(buf, axis=0)
    ph = np.angle(X[::q][:L // 2 + 1])                           # bin m*q <-> m*fs/L
    Y = np.fft.rfft(loop, axis=0)
    Z = np.abs(Y) * np.exp(1j * ph)
    Z[0] = 0.0
    if L % 2 == 0:
        Z[-1] = Z[-1].real
    return np.fft.irfft(Z, n=L, axis=0)


def best_rotation(loop: np.ndarray, target: np.ndarray) -> int:
    """Circular shift tau maximising the correlation of loop[(tau+n) mod L] with target[n]."""
    lm = loop.mean(axis=1) if loop.ndim == 2 else loop
    tm = target.mean(axis=1) if target.ndim == 2 else target
    L = len(lm)
    W = min(len(tm), L)
    tm = tm[:W]
    tl = np.concatenate([lm, lm[:W]])
    c = correlate(tl, tm, mode='valid', method='fft')
    return int(np.argmax(c[:L]))


def rotate_loop(loop: np.ndarray, x: np.ndarray, join: int, fs: int, window_s: float = 0.1) -> tuple[np.ndarray, int]:
    """Rotate the loop so its sample 0 best matches the recording at the join (any basis)."""
    L = len(loop)
    W = int(min(L, window_s * fs, len(x) - join))
    tau = best_rotation(loop, x[join:join + W])
    return np.roll(loop, -tau, axis=0), tau


# ------------------------------------------------------------------------------- level

def level_match(loop: np.ndarray, x: np.ndarray, join: int, fs: int, f0: float,
                per_channel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Scale the loop to the recording's RMS at the join (4 periods, clipped to 20-50 ms)."""
    w = int(np.clip(4 * fs / max(f0, 20.0), 0.02 * fs, 0.05 * fs))
    w = int(min(w, len(x) - join))
    ref = x[join:join + w]
    if per_channel:
        g = _rms(ref, axis=0) / _rms(loop, axis=0)
    else:
        g = np.full(loop.shape[1], float(_rms(ref) / _rms(loop)))
    g = np.clip(g, 0.1, 10.0)
    return loop * g, g


# ------------------------------------------------------------------------------- timbre

def _smooth_log_spectrum(P: np.ndarray, fr: np.ndarray, octaves: float = 1 / 3, min_bins: int = 17) -> np.ndarray:
    """Power spectrum smoothed to a broad envelope: at least ``min_bins`` wide (never resolves the
    harmonic fine structure) and a constant fraction of an octave above that."""
    k = np.ones(min_bins | 1) / (min_bins | 1)
    Pp = np.concatenate([np.full(min_bins // 2, P[0]), P, np.full(min_bins // 2, P[-1])])
    P = np.convolve(Pp, k, mode='valid')[:len(fr)]
    lf = np.log2(np.maximum(fr, 20.0))
    grid = np.linspace(lf[1], lf[-1], 600)
    lp = np.interp(grid, lf, 10 * np.log10(P + 1e-20))
    n = max(3, int(round(octaves / (grid[1] - grid[0]))) | 1)
    kk = np.ones(n) / n
    lpp = np.concatenate([np.full(n // 2, lp[0]), lp, np.full(n // 2, lp[-1])])
    lp = np.convolve(lpp, kk, mode='valid')[:len(grid)]
    return 10 ** (np.interp(lf, grid, lp) / 10)


def morph_tail(tail: np.ndarray, loop: np.ndarray, fs: int, max_db: float = 6.0,
               n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, float]:
    """EQ-morph the recording's tail (the M samples before the join) towards the loop's spectrum.

    A per-band gain — loop LTAS over the tail's last ~40 ms, 1/3-octave smoothed, power-neutral,
    clipped to +-max_db — is applied through an STFT with a raised-cosine ramp from 0 (start of
    the tail: untouched) to 1 (the join).  Returns (morphed tail, largest |gain| applied in dB).
    """
    M, C = tail.shape
    if max_db <= 0 or M < 3 * n_fft // 2:
        return tail, 0.0
    w = np.hanning(n_fft + 1)[:-1]
    fr = np.fft.rfftfreq(n_fft, 1 / fs)
    out = np.zeros_like(tail)
    norm = np.zeros(M)
    n_frames = 1 + (M - n_fft) // hop
    applied = 0.0
    for c in range(C):
        lt = np.tile(loop[:, c], max(1, int(np.ceil(2 * n_fft / len(loop))) + 1))
        T = np.mean([np.abs(np.fft.rfft(lt[s:s + n_fft] * w)) ** 2 for s in range(0, len(lt) - n_fft + 1, hop)], axis=0)
        src = [np.abs(np.fft.rfft(tail[s:s + n_fft, c] * w)) ** 2
               for s in range(max(0, M - n_fft - 2 * hop), M - n_fft + 1, hop)]
        S = np.mean(src, axis=0)
        Ts, Ss = _smooth_log_spectrum(T, fr), _smooth_log_spectrum(S, fr)
        G = np.sqrt(Ts / (Ss + 1e-20))
        G = np.clip(G, 10 ** (-max_db / 20), 10 ** (max_db / 20))
        G *= np.sqrt(np.sum(S) / (np.sum(S * G ** 2) + 1e-20))
        applied = max(applied, float(np.max(np.abs(20 * np.log10(G)))))
        logG = np.log(G)
        for i in range(n_frames):
            s0 = i * hop
            ramp = 0.5 - 0.5 * np.cos(np.pi * min(1.0, (s0 + n_fft / 2) / M))
            X = np.fft.rfft(tail[s0:s0 + n_fft, c] * w)
            out[s0:s0 + n_fft, c] += np.fft.irfft(X * np.exp(ramp * logG), n=n_fft) * w
            if c == 0:
                norm[s0:s0 + n_fft] += w ** 2
    good = norm > 1e-3
    out[good] /= norm[good, None]
    out[~good] = tail[~good]
    return out, applied


# ------------------------------------------------------------------------------- splice

def splice(x: np.ndarray, onset: int, join: int, loop: np.ndarray, fs: int, xfade_s: float = 0.01,
           f0: float | None = None, min_periods: float = 1.0, tail: np.ndarray | None = None) -> tuple[np.ndarray, int, int, int]:
    """recording[onset:join] -> raised-cosine cross-fade into loop[L-X:] -> loop.

    ``tail`` optionally replaces the recording's last len(tail) samples before the join (the
    morphed tail).  Returns (out, loop_start, loop_end, X) with SFZ-style inclusive loop_end.
    """
    L = len(loop)
    X = int(xfade_s * fs)
    if f0:
        X = max(X, int(min_periods * fs / f0))
    X = int(max(1, min(X, join - onset, L // 2)))
    rec = x[onset:join].copy()
    if tail is not None and len(tail):
        rec[-len(tail):] = tail
    w = raised_cosine(X)[:, None]
    head = rec[:len(rec) - X]
    xf = rec[len(rec) - X:] * (1 - w) + loop[L - X:] * w
    out = np.concatenate([head, xf, loop], axis=0)
    nf = min(len(out), int(0.002 * fs))
    if nf > 1:
        out[:nf] *= raised_cosine(nf)[:, None]
    loop_start = join - onset
    loop_end = loop_start + L - 1
    return out, loop_start, loop_end, X


# ------------------------------------------------------------------------------- metrics

def _flux(mono: np.ndarray, fs: int, n: int = 1024, hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Positive spectral flux and the sample position of each flux value."""
    _, _, Z = stft(mono, fs, window='hann', nperseg=n, noverlap=n - hop, boundary=None, padded=False)
    mag = np.abs(Z)
    flux = np.maximum(np.diff(mag, axis=1), 0).sum(axis=0) / (mag[:, 1:].sum(axis=0) + 1e-12)
    pos = (np.arange(len(flux)) + 0.5) * hop + n / 2
    return flux, pos


def junction_metrics(out: np.ndarray, x: np.ndarray, onset: int, loop_start: int, X: int, fs: int,
                     flux_floor: float = 0.01) -> dict:
    """How audible is the join?

    dip_db        level of the cross-fade span relative to the recording over the same span
                  (cancellation shows as < 0)
    transient_db  spectral-flux spike at the join relative to the 90th percentile of the flux in
                  the following 300 ms, on the assembled file minus the same on the recording
    score         max(0, -dip) + max(0, transient) / 4   (lower is better; 0 = no worse than the recording)
    """
    join = onset + loop_start
    a0 = max(onset, join - int(0.3 * fs))
    n_after = int(0.4 * fs)
    seg_out = out[a0 - onset: loop_start + n_after]
    seg_org = x[a0: a0 + len(seg_out)]
    seg_out = seg_out[:len(seg_org)]
    dip = 20 * np.log10(_rms(out[loop_start - X: loop_start]) / _rms(x[join - X: join]))

    def spike(sig):
        f, pos = _flux(sig.mean(axis=1), fs)
        p = join - a0
        at = f[np.abs(pos - p) <= 512]
        ref_sel = (pos > p + 512) & (pos < p + 512 + 0.3 * fs)
        ref = np.percentile(f[ref_sel], 90) if ref_sel.sum() > 3 else np.median(f)
        # normalised flux: 0.01 = a 1 % spectral change per hop.  Real sustained notes sit at
        # 0.02-0.17 (SSO trumpet to string section); the floor stops a synthetic, perfectly
        # steady tone from turning a nothing into a ratio of thousands
        ref = max(ref, flux_floor)
        return 20 * np.log10((at.max() if at.size else 0.0) / ref + 1e-9)

    s_out, s_org = spike(seg_out), spike(seg_org)
    tr = float(np.clip(s_out - s_org, -20, 40))
    return dict(score=float(max(0.0, -dip) + max(0.0, tr) / 4.0), dip_db=float(dip), transient_db=tr,
                join_flux_db=float(s_out), orig_flux_db=float(s_org))


# ------------------------------------------------------------------------------- the whole thing

def join(x: np.ndarray, fs: int, onset: int, join_at: int, loop: np.ndarray, f0: float, *,
         basis: str = 'dft', align: str = 'best', xfade_s: float = 0.01, morph: bool = True,
         morph_s: float = 0.25, morph_max_db: float = 6.0) -> tuple[np.ndarray, dict]:
    """Attack -> loop.  Returns (assembled audio, info).

    align : 'phase' (DFT only), 'rotate', 'none', or 'best' — build phase and rotate candidates
            and keep the one with the lower junction score.
    The loop is level-matched at the join, the recording's tail is EQ-morphed towards the loop,
    and the two are spliced with a short cross-fade.  info carries loop_start / loop_end
    (SFZ-style, loop_end inclusive), the alignment used, gains, and the junction metrics.
    """
    if loop.ndim == 1:
        loop = loop[:, None]
    if x.ndim == 1:
        x = x[:, None]
    cands = []
    if align in ('rotate', 'best', 'none'):
        if align == 'none':
            cands.append(('none', loop, 0))
        else:
            lp, tau = rotate_loop(loop, x, join_at, fs)
            cands.append(('rotate', lp, tau))
    h = int(min(len(loop), join_at - onset, len(x) - join_at))
    if align in ('phase', 'best') and basis == 'dft' and h >= 4 * fs / max(f0, 20.0):
        cands.append(('phase', phase_align_loop(loop, x, join_at, half=h), 0))
    if not cands:
        raise ValueError(f'no alignment possible for align={align!r}, basis={basis!r}')

    Mt = int(min(morph_s * fs, 0.7 * (join_at - onset))) if morph else 0
    results = []
    for name, lp, tau in cands:
        lp, g = level_match(lp, x, join_at, fs, f0)
        tail, applied = (None, 0.0)
        if Mt >= int(0.08 * fs):
            tail, applied = morph_tail(x[join_at - Mt: join_at], lp, fs, max_db=morph_max_db)
        out, ls, le, X = splice(x, onset, join_at, lp, fs, xfade_s=xfade_s, f0=f0, tail=tail)
        jm = junction_metrics(out, x, onset, ls, X, fs)
        results.append((jm['score'], name, out, dict(loop_start=ls, loop_end=le, xfade=X, alignment=name,
                                                    rotation=tau, gain=[float(v) for v in g],
                                                    morph_db=applied, junction=jm)))
    results.sort(key=lambda r: r[0])
    _, _, out, info = results[0]
    info['candidates'] = {name: round(sc, 3) for sc, name, _, _ in results}
    return out, info
