"""Low-level DSP helpers: audio I/O, envelopes, BS.1770 loudness, segmentation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal


# --------------------------------------------------------------------------- I/O
def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 array of shape (n, channels)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if target_sr is not None and target_sr != sr:
        import soxr

        x = soxr.resample(x, sr, target_sr, quality="VHQ").astype(np.float32)
        sr = target_sr
    return x, sr


def write_audio(path: str, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.999:
        x = x * (0.999 / peak)
    sf.write(path, x, sr, subtype=subtype)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def db(a: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(a), floor))


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * math.log2(max(f, 1e-9) / 440.0)


NOTE_NAMES = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]


def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def name_to_midi(name: str) -> int:
    """Parse a note name such as 'a#4', 'Db3', 'c-1' to a MIDI number."""
    s = name.strip().lower()
    i = 1
    if i < len(s) and s[i] in "#b":
        i += 1
    pitch = NOTE_NAMES.index(s[:i].replace("db", "c#").replace("eb", "d#").replace("gb", "f#")
                             .replace("ab", "g#").replace("bb", "a#")) if s[:i] in NOTE_NAMES else None
    if pitch is None:
        flats = {"db": 1, "eb": 3, "gb": 6, "ab": 8, "bb": 10, "cb": 11, "fb": 4}
        pitch = flats[s[:i]]
    octave = int(s[i:])
    return (octave + 1) * 12 + pitch


# ------------------------------------------------------------------ envelopes
def rms_envelope(x: np.ndarray, sr: int, win: float = 0.01, hop: float = 0.0025) -> tuple[np.ndarray, np.ndarray]:
    """Frame RMS in dB. Returns (times_in_samples, env_db)."""
    m = to_mono(x)
    w = max(8, int(round(win * sr)))
    h = max(1, int(round(hop * sr)))
    n_frames = max(1, 1 + (len(m) - w) // h)
    idx = np.arange(n_frames) * h
    frames = np.lib.stride_tricks.as_strided(
        m, shape=(n_frames, w), strides=(m.strides[0] * h, m.strides[0]), writeable=False
    )
    env = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-20)
    return idx + w // 2, db(env)


def smooth(y: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return y
    k = np.ones(n) / n
    ypad = np.concatenate([np.full(n // 2, y[0]), y, np.full(n - n // 2 - 1, y[-1])])
    return np.convolve(ypad, k, mode="valid")


# ------------------------------------------------------------ BS.1770 loudness
def k_weighting_sos(sr: int) -> np.ndarray:
    """K-weighting (ITU-R BS.1770-4) as second-order sections for any sample rate."""
    # Stage 1: high shelf (+4 dB above ~1.7 kHz). Parameters as in pyloudnorm.
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.499666774155
    a0 = 1.0 + K / Q + K * K
    b = [(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a = [1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0]
    s1 = b + a
    # Stage 2: high pass (~38 Hz)
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / sr)
    d = 1.0 + K / Q + K * K
    b = [1.0, -2.0, 1.0]
    a = [1.0, 2.0 * (K * K - 1.0) / d, (1.0 - K / Q + K * K) / d]
    s2 = b + a
    return np.array([s1, s2])


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    sos = k_weighting_sos(sr)
    return signal.sosfilt(sos, x, axis=0)


def loudness_contour(x: np.ndarray, sr: int, block: float = 0.4, hop: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Momentary (400 ms) K-weighted loudness contour in LUFS. Returns (times_s, lufs)."""
    if x.ndim == 1:
        x = x[:, None]
    y = k_weight(x.astype(np.float64), sr)
    w = int(round(block * sr))
    h = int(round(hop * sr))
    n = max(1, 1 + (len(y) - w) // h)
    out = np.empty(n)
    for i in range(n):
        seg = y[i * h : i * h + w]
        ms = np.mean(seg ** 2, axis=0).sum()  # channel weights 1.0 for L/R
        out[i] = -0.691 + 10.0 * np.log10(ms + 1e-20)
    t = (np.arange(n) * h + w / 2) / sr
    return t, out


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness (BS.1770-4) in LUFS."""
    t, l = loudness_contour(x, sr, block=0.4, hop=0.1)
    if l.size == 0:
        return -np.inf
    ms = 10 ** ((l + 0.691) / 10.0)
    keep = l > -70.0
    if not keep.any():
        return -np.inf
    rel = -0.691 + 10 * np.log10(ms[keep].mean()) - 10.0
    keep2 = keep & (l > rel)
    if not keep2.any():
        return -np.inf
    return float(-0.691 + 10 * np.log10(ms[keep2].mean()))


# --------------------------------------------------------------- segmentation
@dataclass
class Segmentation:
    onset: int            # sample index where the note starts (a hair before the rise)
    peak: int             # sample index of the envelope peak
    attack_end: int       # first sample of usable quasi-stationary material
    release_onset: int    # start of the terminal decay (== usable end for decaying notes)
    end: int              # last non-silent sample
    peak_db: float
    body_slope_db_s: float  # linear fit slope of the dB envelope over the body
    klass: str            # 'sustain' | 'decay' | 'oneshot'
    env_t: np.ndarray     # sample positions of the envelope frames
    env_db: np.ndarray


def segment_note(x: np.ndarray, sr: int, f0_hint: float | None = None, min_periods: float = 16.0,
                 force_loop: bool = False) -> Segmentation:
    """Split a sampled note into attack / body / release and classify it."""
    t, e = rms_envelope(x, sr, win=0.01, hop=0.0025)
    e_s = smooth(e, 9)  # ~22 ms smoothing
    peak_i = int(np.argmax(e_s))
    peak_db = float(e_s[peak_i])
    thr = peak_db - 45.0
    above = np.where(e_s > thr)[0]
    first = int(above[0]) if above.size else 0
    last = int(above[-1]) if above.size else len(e_s) - 1
    # backtrack the onset to the local minimum / -60 dB point before the rise
    j = first
    while j > 0 and e_s[j - 1] < e_s[j] and e_s[j - 1] > peak_db - 80:
        j -= 1
    onset = int(max(0, t[j] - int(0.005 * sr)))
    end = int(min(len(x) - 1, t[last] + int(0.02 * sr)))

    # attack end: the onset transient is over once the level is within 15 dB of the peak and the
    # envelope slope over 40 ms is moderate (|slope| < 60 dB/s) - this also handles swelling notes
    # whose envelope maximum comes late.
    hop_s = (t[1] - t[0]) / sr if len(t) > 1 else 0.0025
    win = max(1, int(round(0.04 / hop_s)))
    ae = None
    for i in range(first, len(e_s) - win):
        slope = (e_s[i + win] - e_s[i]) / (win * hop_s)
        if e_s[i] >= peak_db - 15.0 and abs(slope) < 60.0:
            ae = i
            break
    if ae is None:
        ae = peak_i
    ae = max(ae, first + int(round(0.02 / hop_s)))
    if ae < peak_i and e_s[peak_i] - e_s[ae] > 6.0 and (t[peak_i] - t[ae]) / sr < 0.15:
        ae = peak_i  # a genuine short attack: start after the peak
    ae = ae + int(round(0.02 / hop_s))
    attack_end = int(t[min(ae, len(t) - 1)])

    # release onset: sustain level = median of the middle of the body; release starts
    # at the last frame above (sustain - 4 dB) provided the envelope then falls.
    body = e_s[ae:last + 1]
    if body.size > 4:
        mid = body[len(body) // 4 : max(len(body) // 4 + 1, 3 * len(body) // 4)]
        sus_db = float(np.median(mid))
        cand = np.where(body > sus_db - 4.0)[0]
        ro = ae + int(cand[-1]) if cand.size else last
    else:
        sus_db = peak_db
        ro = last
    # for the release onset make sure we back up to where the fall really begins
    release_onset = int(t[min(ro, len(t) - 1)])

    # classify by the slope of a linear fit over the body (dB per second)
    bi = np.arange(ae, max(ae + 2, ro + 1))
    bi = bi[bi < len(e_s)]
    if bi.size >= 2:
        tt = t[bi] / sr
        A = np.vstack([tt, np.ones_like(tt)]).T
        slope, _ = np.linalg.lstsq(A, e_s[bi], rcond=None)[0]
    else:
        slope = 0.0
    body_dur = (release_onset - attack_end) / sr
    min_body = max(0.3, min_periods / f0_hint) if f0_hint else 0.3
    attack_time = (t[peak_i] - onset) / sr
    # how far the note falls within the first two seconds after the attack (a decaying note keeps
    # falling; a sustained one with a diminuendo falls slowly). Short notes: up to the release onset.
    ae_c = min(ae, len(e_s) - 1)
    two_s = int(round(2.0 / hop_s))
    win_end = ae_c + two_s if (t[last] - t[ae_c]) / sr > 2.5 else max(ae_c + 1, min(ro, len(e_s) - 1) + 1)
    win_end = max(ae_c + 1, min(win_end, len(e_s)))
    total_drop = float(e_s[ae_c] - e_s[ae_c: win_end].min())
    if body_dur < min_body and not force_loop:
        klass = "oneshot"
    elif (attack_time < 0.12 and slope < -2.0 and total_drop >= 6.0) or (slope < -12.0 and total_drop >= 15.0):
        klass = "decay"  # excited once, then rings down (piano, harp, mallets, plucked strings)
    else:
        klass = "sustain"
    if klass == "decay":
        # for decaying notes the usable body runs until the level drops 50 dB below peak
        lim = np.where(e_s[peak_i:] < peak_db - 50.0)[0]
        usable = int(t[peak_i + lim[0]]) if lim.size else end
        release_onset = int(min(end, usable))
        if ((release_onset - onset) / sr < 1.0 or slope < -20.0) and not force_loop:
            klass = "oneshot"  # a short or fast-decaying hit: nothing to gain from looping
    return Segmentation(onset, int(t[peak_i]), attack_end, release_onset, end, peak_db,
                        float(slope), klass, t, e_s)


# ----------------------------------------------------------------- misc utils
def raised_cosine(n: int) -> np.ndarray:
    """0 -> 1 raised-cosine ramp of length n."""
    if n <= 1:
        return np.ones(max(n, 0))
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / (n - 1))


def next_pow2(n: int) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(1, n)))))


def lpf2_mag2(f: np.ndarray, fc: float) -> np.ndarray:
    """|H|^2 of a 2-pole (Butterworth-like) low-pass at cutoff fc (sfizz lpf_2p, resonance 0)."""
    r = (np.asarray(f, dtype=np.float64) / max(fc, 1.0)) ** 4
    return 1.0 / (1.0 + r)
