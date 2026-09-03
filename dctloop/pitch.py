"""dctloop.pitch — fundamental frequency of a sustained note.

Three steps, each one needed by the next:

* ``note_from_name``  the note a sample file claims to be ('trumpet-a#4.wav' -> 466.16 Hz), used
                      only to narrow the pitch search;
* ``estimate_f0``     pYIN per channel (coarse — pYIN is quantised to a fraction of a semitone);
* ``refine_f0``       parabolic interpolation of the first harmonic peaks of one long spectrum,
                      good to ~0.01 cent.  The integer-period loop fit needs this: at a 1.5 s
                      loop one grid bin is 0.67 Hz, and pYIN's 10-cent steps are 3 Hz at 466 Hz.
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy.fft import rfft
from scipy.signal.windows import get_window

_NOTE_RE = re.compile(r'(?<![a-z])([a-g])(#|b)?(-?\d)(?![0-9])', re.I)
_SEMITONE = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}


def note_from_name(name: str) -> float | None:
    """Frequency (A4 = 440) of the last note token ('a#4', 'c3', 'bb2') in a file name, or None."""
    m = None
    for m in _NOTE_RE.finditer(name):
        pass
    if m is None:
        return None
    letter, acc, octave = m.group(1).lower(), m.group(2), int(m.group(3))
    midi = 12 * (octave + 1) + _SEMITONE[letter] + (1 if acc == '#' else -1 if acc == 'b' else 0)
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def estimate_f0(seg: np.ndarray, fs: int, hint: float | None = None) -> np.ndarray:
    """Median pYIN f0 per channel in Hz (NaN where nothing is voiced).  ``hint`` narrows the
    search to +-25 % and makes it finer."""
    import librosa
    if seg.ndim == 1:
        seg = seg[:, None]
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
    Hann-windowed spectrum of the whole segment, each harmonic searched within +-``tol`` of
    h * f0 and weighted by its energy."""
    if seg.ndim == 1:
        seg = seg[:, None]
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


def _power_spectrum(seg: np.ndarray, fs: int) -> tuple[np.ndarray, int]:
    """Hann-windowed power spectrum of the channel mean, and the transform length."""
    N = len(seg)
    w = get_window('hann', N, fftbins=True)
    mono = seg.mean(axis=1) if seg.ndim == 2 else seg
    return np.abs(rfft(mono * w)) ** 2, N


def _line_prominence(P: np.ndarray, N: int, fs: int, freq: float, rel: float = 0.012,
                     ctx: float = 0.25) -> float:
    """Energy in a narrow band at ``freq`` over the median band energy of its neighbourhood
    (+-``ctx`` octaves, excluding the line itself): 1 means "nothing there but noise floor"."""
    bw = max(2, int(freq * rel * N / fs))
    k = int(freq * N / fs)
    if k - bw < 1 or k + bw >= len(P):
        return 0.0
    peak = P[k - bw:k + bw + 1].sum()
    lo, hi = max(1, int(freq * 2 ** -ctx * N / fs)), min(len(P) - 1, int(freq * 2 ** ctx * N / fs))
    if hi - lo < 8 * bw:
        return 0.0
    context = np.concatenate([P[lo:k - 2 * bw], P[k + 2 * bw:hi]])
    if not len(context):
        return 0.0
    return float(peak / (np.median(context) * (2 * bw + 1) + 1e-30))


def harmonic_evidence(seg: np.ndarray, fs: int, f0: float, nharm: int = 10,
                      rel: float = 0.012) -> dict:
    """Three cheap tests of whether ``f0`` is really the fundamental of ``seg``:

    * ``frac``      the fraction of spectral energy (up to harmonic ``nharm``) that lies on the
                    harmonic series of f0.  Low when f0 is simply the wrong note, or the sound
                    is unpitched.
    * ``sub_prom``  prominence of the lines at f0/2 and f0/3 over their local noise floor.
                    Large when the true fundamental is an octave (or a twelfth) *below* f0.
                    Measured against the noise floor, not the strongest partial, so a weak
                    fundamental (a trumpet's, 16 dB below its second harmonic) still counts.
    * ``odd_even``  median prominence of the odd harmonics over that of the even ones.  Near
                    zero when the odd lines are empty, i.e. the true fundamental is an octave
                    *above* f0.
    """
    if seg.ndim == 1:
        seg = seg[:, None]
    P, N = _power_spectrum(seg, fs)
    f = np.arange(len(P)) * fs / N
    hi = min(0.45 * fs, f0 * (nharm + 0.5))
    band = (f >= 0.5 * f0) & (f < hi)
    if band.sum() < 10:
        return dict(frac=0.0, sub_prom=0.0, odd_even=0.0)
    ratio = f[band] / f0
    on_harmonic = np.abs(ratio - np.round(ratio)) <= rel * np.round(np.maximum(ratio, 1))
    frac = float(P[band][on_harmonic].sum() / (P[band].sum() + 1e-30))
    sub = [_line_prominence(P, N, fs, f0 / d, rel) for d in (2, 3) if f0 / d >= 25.0]
    odd = [_line_prominence(P, N, fs, h * f0, rel) for h in range(1, nharm, 2) if h * f0 < 0.45 * fs]
    even = [_line_prominence(P, N, fs, h * f0, rel) for h in range(2, nharm + 1, 2) if h * f0 < 0.45 * fs]
    odd_even = float(np.median(odd) / (np.median(even) + 1e-30)) if odd and even else 1.0
    return dict(frac=frac, sub_prom=float(max(sub)) if sub else 0.0, odd_even=odd_even)


def hint_is_trustworthy(seg: np.ndarray, fs: int, f0: float, min_frac: float = 0.40,
                        max_sub_prom: float = 30.0, min_odd_even: float = 0.02) -> tuple[bool, dict]:
    """Does the spectrum bear out a pitch taken from a file name?  Returns (verdict, evidence).

    Thresholds are set from the SSO library with a wide margin: correctly named notes score
    frac 0.89-1.00, sub_prom 0.9-7.5 and odd_even 0.17-1.48, while a name wrong by a semitone,
    a fifth or an octave, or an unpitched sample, fails at least one test by orders of
    magnitude.  ``flute-a4.wav`` is really 884 Hz and is caught by ``odd_even``.
    """
    ev = harmonic_evidence(seg, fs, f0)
    ok = (ev['frac'] >= min_frac and ev['sub_prom'] <= max_sub_prom
          and ev['odd_even'] >= min_odd_even)
    return ok, ev


def f0_per_channel(seg: np.ndarray, fs: int, f0: float | list | None = None,
                   hint: float | None = None, use_hint: bool = True,
                   detail: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    """Per-channel f0 in Hz.

    ``f0`` given            : used as is (one value per channel), or refined per channel from a
                              single value so that stereo detune is still measured.
    ``hint`` given          : refined straight from the hint (e.g. the note in the file name)
                              and accepted if the spectrum bears it out (``hint_is_trustworthy``).
                              This is the fast path: the refinement costs ~7 ms against pYIN's
                              1.2-1.9 s, and converges to the same value to within 0.0001 cent
                              whenever the hint is within about half a semitone.
    otherwise / hint failed : pYIN, then the same refinement.

    ``use_hint=False`` forces pYIN.  With ``detail=True`` returns (f0, info) where info names
    the ``method`` used ('given', 'hint', 'pyin') and carries the hint's ``evidence``.
    """
    if seg.ndim == 1:
        seg = seg[:, None]
    C = seg.shape[1]
    info: dict = {}

    def done(arr, method, **kw):
        info.update(method=method, **kw)
        return (arr, info) if detail else arr

    if f0 is not None:
        arr = np.atleast_1d(np.asarray(f0, dtype=float))
        return done(arr if arr.size == C else refine_f0(seg, fs, float(arr[0])), 'given')

    rejected = False
    if hint and use_hint:
        cand = refine_f0(seg, fs, hint)
        ok, ev = hint_is_trustworthy(seg, fs, float(np.exp(np.mean(np.log(cand)))))
        if ok:
            return done(cand, 'hint', evidence=ev)
        info['evidence'] = ev
        info['hint_rejected'] = float(hint)
        rejected = True

    # a hint the spectrum contradicts must not narrow pYIN's search either: the note name is
    # wrong (SSO's flute-a4.wav is really 884 Hz), so search the full range instead
    coarse = estimate_f0(seg, fs, None if rejected else hint)
    good = coarse[np.isfinite(coarse)]
    if not good.size:
        if hint:
            return done(refine_f0(seg, fs, hint), 'hint_unvoiced')
        return done(np.full(C, np.nan), 'none')
    return done(refine_f0(seg, fs, float(np.exp(np.mean(np.log(good))))), 'pyin')
