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


def f0_per_channel(seg: np.ndarray, fs: int, f0: float | list | None = None,
                   hint: float | None = None) -> np.ndarray:
    """Per-channel f0 in Hz: ``f0`` as given (one value or one per channel), else pYIN + refine.
    A single given value is still refined per channel so that stereo detune is measured."""
    if seg.ndim == 1:
        seg = seg[:, None]
    C = seg.shape[1]
    if f0 is not None:
        arr = np.atleast_1d(np.asarray(f0, dtype=float))
        if arr.size == C:
            return arr
        return refine_f0(seg, fs, float(arr[0]))
    coarse = estimate_f0(seg, fs, hint)
    good = coarse[np.isfinite(coarse)]
    if not good.size:
        if hint:
            return refine_f0(seg, fs, hint)
        return np.full(C, np.nan)
    return refine_f0(seg, fs, float(np.exp(np.mean(np.log(good)))))
