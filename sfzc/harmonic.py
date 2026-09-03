"""Harmonic (sinusoidal) + residual analysis and additive resynthesis.

The analysis is a harmonic-locked peak picker on a zero-phase Hann STFT
(sms-tools style): for every frame and harmonic index k we look for the
spectral peak closest to k*f0*sqrt(1+B*k^2), refine it with parabolic
interpolation, and record per-channel amplitude and phase at the shared
frequency.  The residual is the magnitude spectrogram with harmonic main lobes
masked out (interpolated across), which is what the stochastic synthesis uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import next_pow2, to_mono


# ----------------------------------------------------------------- f0 tracking
def track_f0(x: np.ndarray, sr: int, fmin: float = 27.0, fmax: float = 4300.0,
             hint_hz: float | None = None, hop_s: float = 0.0116, resolution: float = 0.1):
    """pYIN f0 track. Returns (centers_in_samples_at_sr, f0_hz (nan when unvoiced), voiced_prob)."""
    import librosa

    m = to_mono(x).astype(np.float32)
    sr_a = 22050
    if sr != sr_a:
        m = librosa.resample(m, orig_sr=sr, target_sr=sr_a, res_type="soxr_hq")
    if hint_hz is not None:
        fmin, fmax = hint_hz / 1.6, hint_hz * 1.6
    fmin = max(fmin, 20.0)
    fmax = min(fmax, sr_a / 2 - 100)
    frame_length = 4096 if fmin < 45 else 2048
    hop = max(64, int(round(hop_s * sr_a)))
    f0, vflag, vprob = librosa.pyin(m, fmin=fmin, fmax=fmax, sr=sr_a, frame_length=frame_length,
                                    hop_length=hop, fill_na=np.nan, center=True, resolution=resolution)
    centers = (np.arange(len(f0)) * hop) * (sr / sr_a)
    return centers, f0, vprob


def estimate_f0_spectral(x: np.ndarray, sr: int, n0: int, n1: int, fmin: float = 27.0, fmax: float = 4300.0,
                         hint_hz: float | None = None) -> tuple[float, float]:
    """Fast f0 estimate by harmonic summation of log magnitudes on the averaged body spectrum.

    Returns (f0_hz, salience) where salience is the score margin (>~3 means clearly pitched).
    """
    P, fr = _avg_spectrum(x, sr, n0, n1, nfft=16384)
    ldb = 10 * np.log10(P + 1e-20)
    ldb = ldb - np.median(ldb)
    if hint_hz is not None:
        fmin, fmax = hint_hz / 1.6, hint_hz * 1.6
    fmin = max(fmin, 20.0)
    fmax = min(fmax, sr / 4)
    cands = np.geomspace(fmin, fmax, int(np.log2(fmax / fmin) * 600) + 1)  # 2-cent steps
    K = 12
    ks = np.arange(1, K + 1)
    wk = 1.0 / ks  # the true fundamental carries the strongest low harmonics; sub-octaves do not
    fk = cands[:, None] * ks[None, :]
    idx = np.clip(np.round(fk / (fr[1] - fr[0])).astype(int), 0, len(ldb) - 1)
    # take the max over +-1 bin to be tolerant of slight inharmonicity
    vals = np.maximum.reduce([ldb[np.clip(idx + d, 0, len(ldb) - 1)] for d in (-1, 0, 1)])
    valid = fk < sr / 2
    score = (np.where(valid, vals, 0.0) * wk[None, :]).sum(axis=1) / (valid * wk[None, :]).sum(axis=1)
    j = int(np.argmax(score))
    salience = float(score[j] - np.median(score))
    f0 = float(cands[j])
    if hint_hz is None:
        f0 = refine_octave(x, sr, f0, n0, n1)
    return f0, salience


def _avg_spectrum(x: np.ndarray, sr: int, n0: int, n1: int, nfft: int = 8192):
    m = to_mono(x)
    seg = m[max(0, n0): max(0, n1)]
    if len(seg) < nfft:
        seg = np.pad(seg, (0, nfft - len(seg)))
    hop = nfft // 2
    w = np.hanning(nfft)
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for s in range(0, len(seg) - nfft + 1, hop):
        acc += np.abs(np.fft.rfft(seg[s:s + nfft] * w)) ** 2
        cnt += 1
    return acc / max(cnt, 1), np.fft.rfftfreq(nfft, 1 / sr)


def refine_octave(x: np.ndarray, sr: int, f0: float, n0: int, n1: int) -> float:
    """Fix octave errors of an f0 estimate using the averaged body spectrum."""
    P, fr = _avg_spectrum(x, sr, n0, n1)

    def band_energy(f):
        if f <= 0 or f >= sr / 2:
            return 0.0
        i = int(round(f / (sr / 2) * (len(P) - 1)))
        lo, hi = max(0, i - 2), min(len(P), i + 3)
        return float(P[lo:hi].max())

    for _ in range(2):
        # is the true fundamental an octave lower?  odd multiples of f0/2 must carry energy
        half = f0 / 2
        odd = sum(band_energy(k * half) for k in range(1, 16, 2))
        even = sum(band_energy(k * half) for k in range(2, 17, 2))
        if odd > 0.08 * even and half >= 25.0:
            f0 = half
            continue
        # is it an octave higher? odd multiples of f0 all weak compared to even ones
        odd = sum(band_energy(k * f0) for k in range(1, 12, 2))
        even = sum(band_energy(k * f0) for k in range(2, 13, 2))
        if odd < 0.003 * even and 2 * f0 < sr / 4:
            f0 = 2 * f0
            continue
        break
    return f0


# -------------------------------------------------------------------- analysis
@dataclass
class HarmonicModel:
    sr: int
    hop: int
    M: int
    nfft: int
    centers: np.ndarray          # (F,) frame centers in samples
    f0: np.ndarray               # (C, F) refined f0 per channel and frame
    freq: np.ndarray             # (C, K, F) partial frequency (Hz) per channel
    amp: np.ndarray              # (C, K, F) linear amplitude
    phase: np.ndarray            # (C, K, F) phase at the frame center (rad)
    detected: np.ndarray         # (C, K, F) bool
    B: float
    harmonicity: np.ndarray      # (F,) harmonic power / total power (channel mean)
    resid_mag: np.ndarray        # (2, nbins, Fr) residual magnitude (mid, side), STFT units
    resid_freqs: np.ndarray      # (nbins,)
    resid_centers: np.ndarray    # (Fr,) residual frame centers (samples)
    resid_win_sq_sum: float      # sum of squared residual analysis window
    win_sum: float
    win_sq_sum: float
    f0_nominal: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return self.freq.shape[1]

    @property
    def C(self) -> int:
        return self.amp.shape[0]

    @property
    def F(self) -> int:
        return len(self.centers)

    @property
    def freq_mean(self) -> np.ndarray:
        return self.freq.mean(axis=0)

    @property
    def amp_mean(self) -> np.ndarray:
        return self.amp.mean(axis=0)

    def frame_at(self, n: int) -> int:
        """Index of the frame whose center is nearest to sample n."""
        return int(np.clip(np.searchsorted(self.centers, n), 0, self.F - 1))


def _interp_f0_track(centers_src, f0_src, centers_dst, f0_default):
    f0 = np.array(f0_src, dtype=np.float64)
    good = np.isfinite(f0) & (f0 > 0)
    if good.sum() < 2:
        return np.full(len(centers_dst), f0_default)
    return np.interp(centers_dst, centers_src[good], f0[good])


def _zero_phase_frames(x: np.ndarray, centers: np.ndarray, w: np.ndarray, nfft: int) -> np.ndarray:
    """Zero-phase windowed spectra (F, nbins, C) of frames centred at `centers`."""
    M = len(w)
    half = M // 2
    C = x.shape[1]
    xpad = np.concatenate([np.zeros((half, C)), x, np.zeros((M, C))], axis=0)
    out = np.empty((len(centers), nfft // 2 + 1, C), dtype=np.complex64)
    buf = np.zeros((nfft, C))
    for i, c in enumerate(centers):
        fr = xpad[c: c + M] * w[:, None]
        buf[:] = 0
        buf[: M - half] = fr[half:]
        buf[nfft - half:] = fr[:half]
        out[i] = np.fft.rfft(buf, axis=0)
    return out


def _lobe_half_width(w: np.ndarray, nfft: int, level_db: float = -70.0) -> float:
    W = np.abs(np.fft.rfft(w, 64 * nfft))
    W = 20 * np.log10(W / W[0] + 1e-12)
    below = np.where(W < level_db)[0]
    return (below[0] / 64.0) if below.size else 4.0 * nfft / len(w)


def analyze(x: np.ndarray, sr: int, f0_track: tuple[np.ndarray, np.ndarray], f0_nominal: float,
            K_max: int = 120, B: float = 0.0, hop: int | None = None, n_start: int = 0,
            n_end: int | None = None, periods: float = 10.0, min_win_s: float = 0.02,
            max_win_s: float = 0.12, residual: bool = True, prune_db: float = -60.0) -> HarmonicModel:
    """Harmonic-locked sinusoidal analysis, performed independently for every channel.

    Channels of "stereo" samples often carry different mixtures of slightly detuned components
    (pseudo-stereo widening, spaced microphones), so each channel gets its own frequency,
    amplitude and phase tracks.  Only the loop points are shared.
    """
    from scipy.signal.windows import blackmanharris

    if x.ndim == 1:
        x = x[:, None]
    n_end = len(x) if n_end is None else n_end
    C = x.shape[1]
    M = int(round(periods * sr / f0_nominal))
    M = int(np.clip(M, min_win_s * sr, max_win_s * sr))
    M = max(M, int(round(6 * sr / f0_nominal)))  # never fewer than 6 periods
    if M % 2 == 0:
        M += 1
    if hop is None:
        hop = int(min(256, max(64, M // 4)))
    nfft = next_pow2(2 * M)
    w = blackmanharris(M, sym=True)  # -92 dB sidelobes: keeps partial leakage out of the residual
    win_sum, win_sq = float(w.sum()), float((w ** 2).sum())
    half = M // 2
    pad_ratio = nfft / M
    bin_hz = sr / nfft
    lobe_bins = _lobe_half_width(w, nfft)

    centers = np.arange(n_start + half, n_end - half, hop)
    if len(centers) == 0:
        centers = np.array([n_start + half])
    F = len(centers)
    f0_frames = _interp_f0_track(f0_track[0], f0_track[1], centers, f0_nominal)
    K = int(min(K_max, np.floor((sr / 2 - 200.0) / f0_nominal)))
    K = max(K, 1)
    ks = np.arange(1, K + 1)
    inh = np.sqrt(1.0 + B * ks ** 2)

    freq = np.zeros((C, K, F))
    amp = np.zeros((C, K, F))
    phase = np.zeros((C, K, F))
    detected = np.zeros((C, K, F), dtype=bool)
    harmonicity = np.zeros((C, F))
    f0_ref = np.zeros((C, F))
    nb = nfft // 2 + 1
    spectra = _zero_phase_frames(x, centers, w, nfft)  # (F, nb, C)
    glob_max_db = 20 * np.log10(np.abs(spectra).max() + 1e-12)
    thr_db = max(glob_max_db - 96.0, -140.0)
    xpad = np.concatenate([np.zeros((half, C)), x, np.zeros((M, C))], axis=0)

    # windowed frame power per (frame, channel) from Parseval on the zero-phase spectra
    S2 = np.abs(spectra) ** 2
    frame_pow = (S2[:, 0] + 2 * S2[:, 1:-1].sum(axis=1) + S2[:, -1]) / nfft / win_sq   # (F, C)
    for ch in range(C):
        f0_prev = None
        for i, c in enumerate(centers):
            X = spectra[i, :, ch].astype(np.complex128)
            mag_db = 20 * np.log10(np.abs(X) + 1e-12)
            f0_i = f0_prev if f0_prev is not None else f0_frames[i]
            if f0_prev is not None and abs(f0_frames[i] / f0_prev - 1) > 0.06:
                f0_i = f0_frames[i]  # the track says we drifted too far: re-anchor
            for it in range(2):
                exp_f = ks * f0_i * inh
                spacing_bins = f0_i / bin_hz
                hw = max(int(np.ceil(0.35 * spacing_bins)), int(np.ceil(1.5 * pad_ratio)))
                b0 = np.round(exp_f / bin_hz).astype(int)
                idx = np.clip(b0[:, None] + np.arange(-hw, hw + 1)[None, :], 1, nb - 2)   # (K, 2hw+1)
                j = idx[np.arange(K), np.argmax(mag_db[idx], axis=1)]
                mj = mag_db[j]
                det = (mj >= mag_db[j - 1]) & (mj >= mag_db[j + 1]) & (mj >= thr_db) & (b0 < nb - 2)
                a_, b_, c_ = mag_db[j - 1], mj, mag_db[j + 1]
                den = a_ - 2 * b_ + c_
                p = np.where(den != 0, 0.5 * (a_ - c_) / np.where(den != 0, den, 1.0), 0.0)
                p = np.clip(p, -1, 1)
                fk = np.where(det, (j + p) * bin_hz, 0.0)
                ak = np.where(det, 10 ** ((b_ - 0.25 * (a_ - c_) * p) / 20) * 2.0 / win_sum, 0.0)
                phk = np.where(det, np.angle(X[j]), 0.0)
                if it == 0:
                    sel = det & (ks <= 12)
                    if sel.any():
                        est = fk[sel] / (ks[sel] * inh[sel])
                        wgt = ak[sel] ** 2
                        ok = np.abs(est / f0_i - 1) < 0.04
                        if ok.any():
                            f0_i = float(np.sum(est[ok] * wgt[ok]) / np.sum(wgt[ok]))
            f0_ref[ch, i] = f0_i
            f0_prev = f0_i if det[:3].any() else None
            freq[ch, :, i] = np.where(det, fk, ks * f0_i * inh)
            amp[ch, :, i] = ak
            phase[ch, :, i] = phk
            detected[ch, :, i] = det
            harmonicity[ch, i] = 0.5 * np.sum(ak ** 2) / (frame_pow[i, ch] + 1e-20)

    # frequencies of weak / undetected partials -> smoothed harmonic expectation; then low-pass
    # the amplitude and frequency tracks (zero phase) so that frame-rate estimation jitter does
    # not turn into AM/FM sidebands at synthesis (natural vibrato/tremolo < ~25 Hz is kept)
    from scipy.ndimage import median_filter
    from scipy.signal import butter, sosfiltfilt
    frame_rate = sr / hop
    for ch in range(C):
        a_ch = amp[ch]
        strong = detected[ch] & (a_ch > a_ch.max(axis=0, keepdims=True) * 10 ** (-50 / 20))
        exp_freq = ks[:, None] * f0_ref[ch][None, :] * inh[:, None]
        freq[ch] = np.where(strong, freq[ch], exp_freq)
        if F >= 3:
            freq[ch] = median_filter(freq[ch], size=(1, 3), mode="nearest")
        if F >= 16 and frame_rate > 80:
            sos_f = butter(2, min(25.0, 0.4 * frame_rate / 2) / (frame_rate / 2), output="sos")
            sos_a = butter(2, min(40.0, 0.4 * frame_rate / 2) / (frame_rate / 2), output="sos")
            freq[ch] = sosfiltfilt(sos_f, freq[ch], axis=1)
            amp[ch] = np.maximum(sosfiltfilt(sos_a, amp[ch], axis=1), 0.0)

    # prune partials that never come within `prune_db` of the strongest one (inaudible, and they
    # dominate the cost of every later per-sample operation for low notes)
    peak_k = amp.max(axis=(0, 2))
    keep = peak_k >= peak_k.max() * 10 ** (prune_db / 20)
    keep[: min(3, K)] = True
    ks_kept = ks[keep]
    model = HarmonicModel(sr=sr, hop=hop, M=M, nfft=nfft, centers=centers, f0=f0_ref, freq=freq[:, keep],
                          amp=amp[:, keep], phase=phase[:, keep], detected=detected[:, keep], B=B,
                          harmonicity=harmonicity.mean(axis=0),
                          resid_mag=np.zeros((2, 1, 1), np.float32), resid_freqs=np.zeros(1),
                          resid_centers=np.zeros(1), resid_win_sq_sum=1.0, win_sum=win_sum,
                          win_sq_sum=win_sq, f0_nominal=f0_nominal, extra=dict(harmonic_index=ks_kept.tolist(),
                                                                                K_analysed=int(K)))
    if residual:
        residual_analysis(model, x, n_start, n_end)
    return model


def analyze_peaks(x: np.ndarray, sr: int, n_start: int, n_end: int, body: tuple[int, int],
                  K_max: int = 40, win_s: float = 0.06, hop: int | None = None,
                  residual: bool = True, min_prom_db: float = 18.0) -> HarmonicModel | None:
    """Inharmonic sinusoidal analysis (bells, mallets, plucked metal): track the strongest stable
    spectral peaks of the note instead of a harmonic series.  Returns the same HarmonicModel
    structure (partial index k = peak rank by frequency; f0 = the lowest tracked peak)."""
    from scipy.ndimage import median_filter
    from scipy.signal.windows import blackmanharris

    if x.ndim == 1:
        x = x[:, None]
    C = x.shape[1]
    M = int(win_s * sr) | 1
    nfft = next_pow2(2 * M)
    w = blackmanharris(M, sym=True)
    win_sum, win_sq = float(w.sum()), float((w ** 2).sum())
    half = M // 2
    bin_hz = sr / nfft
    hop = hop or max(64, min(256, M // 4))
    lobe_bins = _lobe_half_width(w, nfft)
    centers = np.arange(n_start + half, max(n_start + half + 1, n_end - half), hop)
    spectra = _zero_phase_frames(x, centers, w, nfft)  # (F, nb, C)
    F, nb = len(centers), nfft // 2 + 1
    # average body spectrum (mid) -> candidate peaks
    sel = (centers >= body[0]) & (centers <= body[1])
    if sel.sum() < 2:
        sel = np.ones(F, bool)
    Pm = (np.abs(spectra[sel].mean(axis=2)) ** 2).mean(axis=0)
    mdb = 10 * np.log10(Pm + 1e-20)
    floor = median_filter(mdb, size=int(max(9, 2 * 200 / bin_hz)) | 1, mode="nearest")
    prom = mdb - floor
    cand = [j for j in range(2, nb - 2) if mdb[j] > mdb[j - 1] and mdb[j] >= mdb[j + 1] and prom[j] >= min_prom_db
            and j * bin_hz > 25.0 and j * bin_hz < sr / 2 - 200]
    cand.sort(key=lambda j: -mdb[j])
    peaks = []
    min_sep = 2.0 * lobe_bins
    for j in cand:
        if all(abs(j - p) >= min_sep for p in peaks):
            peaks.append(j)
        if len(peaks) >= K_max:
            break
    if len(peaks) < 2:
        return None
    peaks.sort()
    pf = np.array([p * bin_hz for p in peaks])
    K = len(pf)
    freq = np.zeros((C, K, F))
    amp = np.zeros((C, K, F))
    phase = np.zeros((C, K, F))
    detected = np.zeros((C, K, F), dtype=bool)
    harmonicity = np.zeros((C, F))
    xpad = np.concatenate([np.zeros((half, C)), x, np.zeros((M, C))], axis=0)
    glob_max_db = 20 * np.log10(np.abs(spectra).max() + 1e-12)
    thr_db = max(glob_max_db - 96.0, -140.0)
    for ch in range(C):
        for i, c in enumerate(centers):
            X = spectra[i, :, ch].astype(np.complex128)
            mag_db = 20 * np.log10(np.abs(X) + 1e-12)
            ak = np.zeros(K)
            for k, f in enumerate(pf):
                b0 = int(round(f / bin_hz))
                hw = max(int(np.ceil(0.015 * f / bin_hz)), int(np.ceil(lobe_bins)))
                lo, hi = max(1, b0 - hw), min(nb - 2, b0 + hw)
                j = int(np.argmax(mag_db[lo: hi + 1])) + lo
                if not (mag_db[j] >= mag_db[j - 1] and mag_db[j] >= mag_db[j + 1]) or mag_db[j] < thr_db:
                    freq[ch, k, i] = f
                    continue
                a, b, cc = mag_db[j - 1], mag_db[j], mag_db[j + 1]
                den = a - 2 * b + cc
                p = float(np.clip(0.5 * (a - cc) / den if den != 0 else 0.0, -1, 1))
                freq[ch, k, i] = (j + p) * bin_hz
                ak[k] = 10 ** ((b - 0.25 * (a - cc) * p) / 20) * 2.0 / win_sum
                amp[ch, k, i] = ak[k]
                phase[ch, k, i] = np.angle(X[j])
                detected[ch, k, i] = True
            tot_pow = np.sum((xpad[c: c + M, ch] * w) ** 2) / win_sq
            harmonicity[ch, i] = 0.5 * np.sum(ak ** 2) / (tot_pow + 1e-20)
    from scipy.signal import butter, sosfiltfilt
    frame_rate = sr / hop
    for ch in range(C):
        if F >= 16 and frame_rate > 80:
            sos_f = butter(2, min(25.0, 0.4 * frame_rate / 2) / (frame_rate / 2), output="sos")
            sos_a = butter(2, min(40.0, 0.4 * frame_rate / 2) / (frame_rate / 2), output="sos")
            freq[ch] = sosfiltfilt(sos_f, freq[ch], axis=1)
            amp[ch] = np.maximum(sosfiltfilt(sos_a, amp[ch], axis=1), 0.0)
    f0_ref = freq[:, 0, :].copy()
    model = HarmonicModel(sr=sr, hop=hop, M=M, nfft=nfft, centers=centers, f0=f0_ref, freq=freq, amp=amp,
                          phase=phase, detected=detected, B=0.0, harmonicity=harmonicity.mean(axis=0),
                          resid_mag=np.zeros((2, 1, 1), np.float32), resid_freqs=np.zeros(1),
                          resid_centers=np.zeros(1), resid_win_sq_sum=1.0, win_sum=win_sum, win_sq_sum=win_sq,
                          f0_nominal=float(pf[0]), extra=dict(mode="peaks", peak_freqs=pf.tolist()))
    if residual:
        residual_analysis(model, x, n_start, n_end)
    return model


def residual_analysis(model: HarmonicModel, x: np.ndarray, n_start: int, n_end: int,
                      win_factor: float = 4.0) -> None:
    """Residual (mid/side) magnitude spectrogram with the harmonic lobes masked out.

    Uses a window `win_factor` times longer than the harmonic analysis so that the between-
    harmonic noise floor is resolved well (the residual needs frequency, not time, resolution).
    """
    from scipy.signal.windows import blackmanharris

    if x.ndim == 1:
        x = x[:, None]
    C = x.shape[1]
    sr = model.sr
    M = int(model.M * win_factor)
    M = min(M, int(0.5 * sr))
    if M % 2 == 0:
        M += 1
    nfft = next_pow2(2 * M)
    hop = max(model.hop, M // 4)
    w = blackmanharris(M, sym=True)
    half = M // 2
    lobe = _lobe_half_width(w, nfft)
    mask_half = int(np.ceil(lobe)) + 1
    bin_hz = sr / nfft
    nb = nfft // 2 + 1
    centers = np.arange(n_start + half, max(n_start + half + 1, n_end - half), hop)
    spectra = _zero_phase_frames(x, centers, w, nfft)  # (Fr, nb, C)
    if C == 2:
        Xm = 0.5 * (spectra[:, :, 0] + spectra[:, :, 1])
        Xs = 0.5 * (spectra[:, :, 0] - spectra[:, :, 1])
    else:
        Xm = spectra[:, :, 0]
        Xs = np.zeros_like(Xm)
    resid = np.zeros((2, nb, len(centers)), dtype=np.float32)
    from scipy.ndimage import median_filter
    floor_size = int(max(4 * (2 * mask_half + 1), model.f0_nominal / bin_hz)) | 1
    for i, c in enumerate(centers):
        # harmonic positions of every channel at this time (nearest analysis frame)
        fi = model.frame_at(c)
        det = model.detected[:, :, fi]
        j_all = np.round(model.freq[:, :, fi][det] / bin_hz).astype(int)
        j_all = j_all[(j_all >= 0) & (j_all < nb)]
        mask = np.zeros(nb + 1, dtype=np.int32)
        np.add.at(mask, np.clip(j_all - mask_half, 0, nb), 1)
        np.add.at(mask, np.clip(j_all + mask_half + 1, 0, nb), -1)
        masked = np.cumsum(mask[:nb]) > 0
        for q, Xq in enumerate((Xm, Xs)):
            mdb = 20 * np.log10(np.abs(Xq[i]) + 1e-12)
            floor = median_filter(mdb, size=floor_size, mode="nearest")   # robust between-lobe floor
            out = np.where(masked, np.minimum(mdb, floor), mdb)
            resid[q, :, i] = (10 ** (out / 20)).astype(np.float32)
    model.resid_mag = resid
    model.resid_freqs = np.fft.rfftfreq(nfft, 1 / sr)
    model.resid_centers = centers
    model.resid_win_sq_sum = float((w ** 2).sum())


def estimate_inharmonicity(model: HarmonicModel, i0: int, i1: int) -> float:
    """Weighted LS estimate of B in f_k = k f0 sqrt(1 + B k^2) over frames i0..i1 (all channels)."""
    ks = np.array(model.extra.get("harmonic_index", list(range(1, model.K + 1))))
    Bs = []
    for ch in range(model.C):
        for i in range(i0, min(i1, model.F)):
            det = model.detected[ch, :, i] & (ks >= 2) & (ks <= 40)
            if det.sum() < 4:
                continue
            f0 = model.f0[ch, i]
            y = (model.freq[ch, det, i] / (ks[det] * f0)) ** 2 - 1.0
            k2 = ks[det].astype(float) ** 2
            wgt = model.amp[ch, det, i] ** 2
            den = np.sum(wgt * k2 ** 2)
            if den > 0:
                Bs.append(np.sum(wgt * y * k2) / den)
    if not Bs:
        return 0.0
    return max(float(np.median(Bs)), 0.0)


# ------------------------------------------------------------------- synthesis
def interp_tracks(model_centers: np.ndarray, tracks: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Linearly interpolate frame tracks (..., F) to samples n_a..n_b-1 -> (..., n_b-n_a).

    Frame centres are uniformly spaced, so this is pure index arithmetic (vectorised over tracks).
    """
    F = len(model_centers)
    if F < 2:
        return np.repeat(tracks[..., :1], n_b - n_a, axis=-1)
    hop = float(model_centers[1] - model_centers[0])
    c0 = float(model_centers[0])
    N = n_b - n_a
    out = np.empty(tracks.shape[:-1] + (N,))
    n = n_a
    while n < n_b:
        pos = (n - c0) / hop
        i0 = int(np.clip(np.floor(pos), 0, F - 2))
        seg_end = min(n_b, int(np.ceil(c0 + (i0 + 1) * hop)))
        if seg_end <= n:
            seg_end = n + 1
        m = seg_end - n
        w = np.clip((np.arange(n, seg_end) - c0) / hop - i0, 0.0, 1.0)
        t0 = tracks[..., i0][..., None]
        out[..., n - n_a: n - n_a + m] = t0 + (tracks[..., i0 + 1][..., None] - t0) * w
        n = seg_end
    return out


def phase_advance(freq: np.ndarray, sr: int) -> np.ndarray:
    """Total phase advance (rad) of each partial over per-sample frequency tracks (..., N)."""
    return 2 * np.pi * freq.sum(axis=-1) / sr


def synth_partials(amp: np.ndarray, freq: np.ndarray, phase0: np.ndarray, sr: int,
                   slope: np.ndarray | None = None, phase_ref: int = 0, chunk: int = 16) -> np.ndarray:
    """Additive synthesis.

    amp: (C, K, N) per-sample amplitudes; freq: (C, K, N) or (K, N) per-sample Hz;
    phase0: (C, K) phase at sample index `phase_ref`; slope: (C, K) or (K,) extra linear phase
    per sample (loop-closing correction).  Returns (N, C).
    """
    C, K, N = amp.shape
    if freq.ndim == 2:
        freq = np.broadcast_to(freq, (C, K, N))
    if slope is not None and slope.ndim == 1:
        slope = np.broadcast_to(slope, (C, K))
    out = np.zeros((N, C))
    n = np.arange(N)
    for c in range(C):
        for k0 in range(0, K, chunk):
            k1 = min(K, k0 + chunk)
            ph = np.cumsum(2 * np.pi * freq[c, k0:k1] / sr, axis=1)
            ph = np.concatenate([np.zeros((k1 - k0, 1)), ph[:, :-1]], axis=1)
            if slope is not None:
                ph = ph + slope[c, k0:k1, None] * n[None, :]
            ph = ph - ph[:, phase_ref: phase_ref + 1] + phase0[c, k0:k1, None]
            out[:, c] += np.sum(amp[c, k0:k1] * np.cos(ph), axis=0)
    return out


def harmonic_centroid(amp: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Power-weighted centroid (Hz) of a harmonic amplitude set. amp: (K, F) or (K,), freq same."""
    p = amp ** 2
    return np.sum(p * freq, axis=0) / (np.sum(p, axis=0) + 1e-20)
