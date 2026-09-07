"""dctjoin.segment — where does the attack end?

A recorded note is split into onset -> attack -> body -> release from its RMS envelope.  The
attack end is the first instant where the level is within ``settle_db`` of the peak and the
envelope slope over 40 ms is moderate, plus a 20 ms margin: the onset transient is over and the
tone is quasi-stationary.  That is the earliest sensible place to hand over to a loop.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def _smooth(y: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(y) < 2:
        return y
    k = np.ones(n) / n
    ypad = np.concatenate([np.full(n // 2, y[0]), y, np.full(n - n // 2 - 1, y[-1])])
    return np.convolve(ypad, k, mode='valid')


def rms_envelope_db(x: np.ndarray, fs: int, win: float = 0.01, hop: float = 0.0025):
    """(frame centres in samples, RMS in dB) of the channel mean."""
    m = _to_mono(x).astype(np.float64)
    w, h = max(8, int(round(win * fs))), max(1, int(round(hop * fs)))
    if len(m) < w:
        return np.array([len(m) // 2]), np.array([20 * np.log10(np.sqrt(np.mean(m ** 2)) + 1e-10)])
    frames = np.lib.stride_tricks.sliding_window_view(m, w)[::h]
    env = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)
    return np.arange(len(frames)) * h + w // 2, 20 * np.log10(env)


@dataclass
class Segments:
    onset: int          # a hair before the rise
    peak: int           # envelope maximum
    attack_end: int     # first quasi-stationary sample: the default join point
    release_onset: int  # where the terminal fall starts: the end of loopable material
    end: int            # last non-silent sample
    peak_db: float

    def seconds(self, fs: int) -> dict:
        d = asdict(self)
        return {k: (v / fs if k != 'peak_db' else v) for k, v in d.items()}


def segment_note(x: np.ndarray, fs: int, settle_db: float = 15.0, max_slope_db_s: float = 60.0) -> Segments:
    t, e = rms_envelope_db(x, fs)
    e_s = _smooth(e, 9)                                    # ~22 ms
    peak_i = int(np.argmax(e_s))
    peak_db = float(e_s[peak_i])
    above = np.where(e_s > peak_db - 45.0)[0]
    first = int(above[0]) if above.size else 0
    last = int(above[-1]) if above.size else len(e_s) - 1
    j = first
    while j > 0 and e_s[j - 1] < e_s[j] and e_s[j - 1] > peak_db - 80:
        j -= 1
    onset = int(max(0, t[j] - int(0.005 * fs)))
    end = int(min(len(x) - 1, t[last] + int(0.02 * fs)))

    hop_s = (t[1] - t[0]) / fs if len(t) > 1 else 0.0025
    win = max(1, int(round(0.04 / hop_s)))
    ae = None
    for i in range(first, len(e_s) - win):
        slope = (e_s[i + win] - e_s[i]) / (win * hop_s)
        if e_s[i] >= peak_db - settle_db and abs(slope) < max_slope_db_s:
            ae = i
            break
    if ae is None:
        ae = peak_i
    ae = max(ae, first + int(round(0.02 / hop_s)))
    if ae < peak_i and e_s[peak_i] - e_s[ae] > 6.0 and (t[peak_i] - t[ae]) / fs < 0.15:
        ae = peak_i                                        # a genuine short attack: start after the peak
    ae = ae + int(round(0.02 / hop_s))
    attack_end = int(t[min(ae, len(t) - 1)])

    body = e_s[ae:last + 1]
    if body.size > 4:
        mid = body[len(body) // 4: max(len(body) // 4 + 1, 3 * len(body) // 4)]
        sus_db = float(np.median(mid))
        cand = np.where(body > sus_db - 4.0)[0]
        ro = ae + int(cand[-1]) if cand.size else last
    else:
        ro = last
    release_onset = int(t[min(ro, len(t) - 1)])
    if release_onset <= attack_end:
        release_onset = end
    return Segments(onset, int(t[peak_i]), attack_end, release_onset, end, peak_db)
