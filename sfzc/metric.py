"""Algorithm B: perceptual recreation / loop quality metric.

score = exp(-(w_spec*D_spec + w_loud*D_loud + w_seam*D_seam + w_var*D_var + w_pitch*D_pitch))

D_spec  : multi-resolution log-spectral (mel + linear STFT) distance, dB
D_loud  : BS.1770 momentary-loudness contour error, LU
D_seam  : seam detectability - excess periodicity of a spectral-flux novelty
          function (and of the amplitude-envelope modulation spectrum) in the
          rendered signal relative to the original
D_var   : temporal micro-variation mismatch (dead-loop detector), log ratio
D_pitch : f0 offset / vibrato depth / vibrato rate mismatch

The default weights are uncalibrated engineering defaults (see calibrate()).
"""
from __future__ import annotations

import numpy as np

from . import dsp

DEFAULT_WEIGHTS = dict(spec=1 / 9.0, loud=1 / 6.0, seam=2.5, var=0.6, pitch=0.12)


def _align(orig: np.ndarray, rend: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(orig), len(rend))
    o = np.zeros((n, orig.shape[1]), np.float32)
    r = np.zeros((n, rend.shape[1]), np.float32)
    o[: len(orig)] = orig
    r[: len(rend)] = rend
    return o, r


_MEL_CACHE: dict = {}


def _mel_fb(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    key = (sr, n_fft, n_mels)
    if key not in _MEL_CACHE:
        import librosa

        _MEL_CACHE[key] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=30, fmax=min(16000, sr / 2))
    return _MEL_CACHE[key]


def _power_stft(m: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """|STFT|^2 (bins, frames) with a Hann window, centred (matches librosa's framing)."""
    x = np.pad(m.astype(np.float32), (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0]))
    w = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    return (np.abs(np.fft.rfft(frames * w, axis=1)) ** 2).T


def _logmel(m: np.ndarray, sr: int, n_fft: int, hop: int, n_mels: int = 96):
    S = _mel_fb(sr, n_fft, n_mels) @ _power_stft(m, n_fft, hop)
    return 10 * np.log10(S + 1e-10)


def _block_avg(L: np.ndarray, n: int) -> np.ndarray:
    """Average a (bands, frames) log spectrogram in the power domain over blocks of n frames."""
    F = L.shape[1] // n * n
    if F == 0:
        return L
    P = 10 ** (L[:, :F] / 10)
    return 10 * np.log10(P.reshape(L.shape[0], -1, n).mean(axis=2) + 1e-10)


def spectral_distance(o: np.ndarray, r: np.ndarray, sr: int) -> float:
    """Frame-weighted mean |dB difference| over mel and linear log spectra (two resolutions).

    The mel term is evaluated both per frame and averaged over 250 ms blocks (so that vibrato
    phase and micro-variation misalignment - inaudible as timbre error - do not dominate).
    """
    om, rm = dsp.to_mono(o), dsp.to_mono(r)
    d_total, w_total = 0.0, 0.0
    for n_fft in (1024, 4096):
        hop = n_fft // 4
        O, R = _logmel(om, sr, n_fft, hop), _logmel(rm, sr, n_fft, hop)
        nb = max(1, int(round(0.25 * sr / hop)))
        Ob, Rb = _block_avg(O, nb), _block_avg(R, nb)
        bw = np.maximum(10 ** ((Ob.max(axis=0) - Ob.max()) / 20), 1e-3)
        db_ = np.abs(np.clip(Ob, -80, None) - np.clip(Rb, -80, None)).mean(axis=0)
        d_total += 2.0 * np.sum(db_ * bw) / np.sum(bw)
        frame_w = 10 ** ((O.max(axis=0) - O.max()) / 20)  # louder frames matter more
        frame_w = np.maximum(frame_w, 1e-3)
        d = np.abs(np.clip(O, -80, None) - np.clip(R, -80, None)).mean(axis=0)
        d_total += np.sum(d * frame_w) / np.sum(frame_w)
        w_total += 2
        # linear-frequency STFT term (captures fine harmonic structure)
        So = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(om, n_fft)[::hop] * np.hanning(n_fft), axis=1))
        Sr = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(rm, n_fft)[::hop] * np.hanning(n_fft), axis=1))
        lo, lr = 20 * np.log10(So + 1e-6), 20 * np.log10(Sr + 1e-6)
        fw = np.maximum(10 ** ((lo.max(axis=1) - lo.max()) / 20), 1e-3)
        dl = np.abs(np.clip(lo, -60, None) - np.clip(lr, -60, None)).mean(axis=1)
        d_total += np.sum(dl * fw) / np.sum(fw)
        w_total += 1
    return float(d_total / w_total)


def loudness_distance(o: np.ndarray, r: np.ndarray, sr: int) -> tuple[float, float]:
    """Mean |LU| difference of momentary loudness contours (after global gain match). Returns (D, gain_db)."""
    lo_i = dsp.integrated_loudness(o, sr)
    lr_i = dsp.integrated_loudness(r, sr)
    gain = float(lo_i - lr_i) if np.isfinite(lo_i) and np.isfinite(lr_i) else 0.0
    _, lo = dsp.loudness_contour(o, sr, block=0.2, hop=0.05)
    _, lr = dsp.loudness_contour(r, sr, block=0.2, hop=0.05)
    n = min(len(lo), len(lr))
    lo, lr = lo[:n], lr[:n] + gain  # envelope *shape* error after global level match
    floor = lo.max() - 55.0
    sel = lo > lo.max() - 50
    if sel.sum() == 0:
        return 0.0, gain
    lo_c, lr_c = np.maximum(lo, floor), np.maximum(lr, floor)
    return float(np.mean(np.abs(lo_c[sel] - lr_c[sel])) + 0.25 * abs(gain)), gain


def _novelty(m: np.ndarray, sr: int, hop: int = 256, n_fft: int = 1024) -> np.ndarray:
    L = _logmel(m, sr, n_fft, hop, n_mels=64)
    d = np.diff(L, axis=1)
    return np.maximum(d, 0).sum(axis=0)


def _periodicity(nov: np.ndarray, hop_s: float, lag_min_s: float, lag_max_s: float,
                 lag_target_s: float | None = None) -> tuple[float, float]:
    """Max normalised autocorrelation of the novelty function in the lag range (and at a target lag)."""
    n = nov - nov.mean()
    if len(n) < 8 or np.std(n) < 1e-9:
        return 0.0, 0.0
    ac = np.correlate(n, n, mode="full")[len(n) - 1:]
    ac = ac / (ac[0] + 1e-12)
    # unbiased-ish normalisation for the shrinking overlap
    counts = np.arange(len(n), 0, -1)
    ac = ac * len(n) / counts
    lmin = max(1, int(lag_min_s / hop_s))
    lmax = min(len(ac) - 1, int(lag_max_s / hop_s))
    if lmax <= lmin:
        return 0.0, 0.0
    r_max = float(np.max(ac[lmin:lmax + 1]))
    r_t = 0.0
    if lag_target_s is not None:
        lt = int(round(lag_target_s / hop_s))
        if lmin <= lt <= lmax:
            r_t = float(np.max(ac[max(lmin, lt - 2): min(lmax, lt + 2) + 1]))
    return r_max, r_t


def _env_modulation_peak(m: np.ndarray, sr: int) -> float:
    """Peak prominence (dB) of the log-envelope modulation spectrum in 0.5-20 Hz."""
    t, e = dsp.rms_envelope(m, sr, win=0.01, hop=0.005)
    if len(e) < 64:
        return 0.0
    e = e - dsp.smooth(e, 101)  # remove slow trend (>0.5 s)
    e = e - e.mean()
    n = 1 << int(np.ceil(np.log2(len(e) * 4)))
    S = np.abs(np.fft.rfft(e * np.hanning(len(e)), n=n)) ** 2
    fr = np.fft.rfftfreq(n, 0.005)
    band = (fr >= 0.5) & (fr <= 20)
    if band.sum() < 4:
        return 0.0
    Sb = S[band]
    return float(10 * np.log10((Sb.max() + 1e-12) / (np.median(Sb) + 1e-12)))


def _pumping_depth(m: np.ndarray, sr: int, start: int, N: int) -> float:
    """Std (dB) of the loop-cycle-folded, 100 ms-smoothed log envelope: periodic level pumping."""
    t, e = dsp.rms_envelope(m, sr, win=0.01, hop=0.0025)
    e = dsp.smooth(e, 41)
    sel = (t >= start) & (t < start + max(4 * N, int(1.5 * sr)))
    if sel.sum() < 16 or N <= 0:
        return 0.0
    ph = ((t[sel] - start) % N) / N
    nb = 32
    bins = np.minimum((ph * nb).astype(int), nb - 1)
    ee = e[sel]
    pat = np.array([ee[bins == i].mean() if np.any(bins == i) else np.nan for i in range(nb)])
    pat = pat[np.isfinite(pat)]
    return float(np.std(pat)) if pat.size > 2 else 0.0


def seam_distance(o: np.ndarray, r: np.ndarray, sr: int, held: tuple[float, float] | None,
                  loop_period_s: float | None, loop_start_s: float | None = None) -> dict:
    om, rm = dsp.to_mono(o), dsp.to_mono(r)
    crop = 0
    if held is not None:
        a, b = int(held[0] * sr), int(held[1] * sr)
        if b - a > sr // 4:
            om, rm = om[a:b], rm[a:b]
            crop = a
    hop = 256
    hop_s = hop / sr
    no, nr = _novelty(om, sr, hop), _novelty(rm, sr, hop)
    # transient-only novelty: remove everything slower than ~50 ms so that content repetition
    # (vibrato, micro-variation) does not count, only click-like events at the seam do
    k = max(3, int(round(0.05 / hop_s)) | 1)
    no_hp = np.maximum(no - dsp.smooth(no, k), 0)
    nr_hp = np.maximum(nr - dsp.smooth(nr, k), 0)
    dur = len(om) / sr
    lag_max = min(4.0, dur / 2.5)
    po, _ = _periodicity(no_hp, hop_s, 0.04, lag_max, loop_period_s)
    pr, prt = _periodicity(nr_hp, hop_s, 0.04, lag_max, loop_period_s)
    excess_ac = max(0.0, pr - po)
    mo, mr = _env_modulation_peak(om, sr), _env_modulation_peak(rm, sr)
    pump_o = pump_r = 0.0
    if loop_period_s is not None and loop_start_s is not None:
        N = int(loop_period_s * sr)
        st = int(loop_start_s * sr) - crop
        pump_o = _pumping_depth(om, sr, max(st, 0), N)
        pump_r = _pumping_depth(rm, sr, max(st, 0), N)
    excess_mod = max(0.0, pump_r - pump_o) / 4.0  # 4 dB of periodic level pumping == 1.0
    # explicit comb measure at the known loop period (diagnostic): novelty at seam times vs elsewhere
    seam_db = 0.0
    if loop_period_s is not None and loop_period_s > 0.02:
        per = loop_period_s / hop_s
        if per >= 2 and len(nr) > 3 * per:
            # novelty frames are centred at frame*hop; the seams sit at loop_start + m*loop_len
            frames = np.arange(len(nr_hp))
            if loop_start_s is not None:
                ph_seam = ((loop_start_s * sr - crop) / hop) % per
                dist = np.abs(((frames - ph_seam + per / 2) % per) - per / 2)
                at = dist < 1.0
                if at.sum() >= 2 and (~at).sum() >= 2:
                    # rank the seam phase against all other phase classes
                    mu = [nr_hp[np.abs(((frames - ph + per / 2) % per) - per / 2) < 1.0].mean()
                          for ph in np.arange(0, per, max(1.0, per / 64))]
                    ref = np.percentile(mu, 90)
                    seam_db = float(20 * np.log10((nr_hp[at].mean() + 1e-9) / (ref + 1e-9)))
            else:
                best = -np.inf
                for ph in np.linspace(0, per, 16, endpoint=False):
                    at = np.abs(((frames - ph + per / 2) % per) - per / 2) < 1.0
                    if at.sum() >= 2 and (~at).sum() >= 2:
                        best = max(best, 20 * np.log10((nr_hp[at].mean() + 1e-9) / (nr_hp[~at].mean() + 1e-9)))
                seam_db = float(best - 6.0) if np.isfinite(best) else 0.0  # max-over-phases bias
    # seam prominence (dB above the loop's typical transient level) is the primary cue
    seam_db = max(seam_db, -20.0)
    # the one-time junction where the recorded attack hands over to the loop: spectral step across
    # loop_start (mean |dB| log-mel difference, 40 ms before vs 40 ms after), in excess of the
    # original's own step at the same instant. An absolute measure, so steady notes are not
    # penalised for tiny steps relative to a near-zero transient floor.
    junction_db = 0.0
    if loop_start_s is not None:
        j0 = int(loop_start_s * sr)
        n40 = int(0.04 * sr)
        if j0 - n40 >= 0 and j0 + n40 <= min(len(o), len(r)):
            def _step(sig):
                m = dsp.to_mono(sig)
                A = _logmel(m[j0 - n40: j0], sr, 1024, 256, n_mels=48).mean(axis=1)
                B = _logmel(m[j0: j0 + n40], sr, 1024, 256, n_mels=48).mean(axis=1)
                lo = max(A.max(), B.max()) - 50.0
                return float(np.mean(np.abs(np.maximum(A, lo) - np.maximum(B, lo))))
            junction_db = float(np.clip(_step(r) - _step(o), -20.0, 40.0))
    D = min(1.5, 0.1 * excess_ac + 0.5 * excess_mod + max(0.0, seam_db - 3.0) / 10.0 + max(0.0, junction_db - 1.5) / 6.0)
    return dict(D_seam=float(D), periodicity_orig=float(po), periodicity_rend=float(pr),
                periodicity_rend_at_loop=float(prt), env_mod_peak_orig_db=float(mo), env_mod_peak_rend_db=float(mr),
                pumping_orig_db=float(pump_o), pumping_rend_db=float(pump_r), seam_prominence_db=float(seam_db),
                junction_db=float(junction_db))


def variation_distance(o: np.ndarray, r: np.ndarray, sr: int, held: tuple[float, float] | None) -> dict:
    om, rm = dsp.to_mono(o), dsp.to_mono(r)
    if held is not None:
        a, b = int(held[0] * sr), int(held[1] * sr)
        if b - a > sr // 4:
            om, rm = om[a:b], rm[a:b]

    def feats(m):
        from scipy.fft import dct

        hop = 512
        P = _power_stft(m, 2048, hop)
        L = 10 * np.log10(_mel_fb(sr, 2048, 64) @ P + 1e-10)
        mf = dct(L, type=2, axis=0, norm="ortho")[1:13]
        fr = np.fft.rfftfreq(2048, 1 / sr)[:, None]
        cen = (P * fr).sum(axis=0) / (P.sum(axis=0) + 1e-12)
        rms = np.sqrt(P.mean(axis=0) / 2048)
        f = np.vstack([mf, np.log(cen + 1)[None], 20 * np.log10(rms + 1e-6)[None]])
        # remove slow trend (~0.5 s) so only micro-variation remains
        k = max(3, int(0.5 * sr / hop) | 1)
        trend = np.stack([dsp.smooth(row, k) for row in f])
        return (f - trend).std(axis=1)

    so, sr_ = feats(om), feats(rm)
    under = np.maximum(0.0, np.log((so + 1e-4) / (sr_ + 1e-4)))
    over = np.maximum(0.0, np.log((sr_ + 1e-4) / (so + 1e-4)))
    D = float(under.mean() + 0.35 * over.mean())
    return dict(D_var=D, var_under=float(under.mean()), var_over=float(over.mean()))


def _hilbert_f0(m: np.ndarray, sr: int, f_track: float, hop_s: float = 0.01) -> np.ndarray:
    """Instantaneous frequency of the partial near f_track (band-pass + analytic signal), sampled every hop_s."""
    from scipy.signal import butter, hilbert, sosfiltfilt

    lo, hi = f_track / 1.06, f_track * 1.06
    sos = butter(4, [lo, min(hi, sr / 2 * 0.98)], btype="band", fs=sr, output="sos")
    an = hilbert(sosfiltfilt(sos, m.astype(np.float64)))
    f = np.diff(np.unwrap(np.angle(an))) * sr / (2 * np.pi)
    env = np.abs(an[1:])
    f = dsp.smooth(f, int(0.005 * sr) | 1)
    hop = max(1, int(hop_s * sr))
    f, env = f[::hop], env[::hop]
    good = env > 0.05 * env.max()                       # ignore near-silent frames
    f = f[good]
    return f[(f > lo * 0.9) & (f < hi * 1.1)]


def pitch_distance(o: np.ndarray, r: np.ndarray, sr: int, held: tuple[float, float] | None,
                   f0_hint: float | None = None, track_hz: float | None = None) -> dict:
    from .harmonic import track_f0

    def stats(x):
        xs = x
        if held is not None:
            a, b = int(held[0] * sr), int(held[1] * sr)
            if b - a > sr // 4:
                xs = x[a:b]
        if track_hz is not None or f0_hint is not None:
            f0 = _hilbert_f0(dsp.to_mono(xs), sr, track_hz or f0_hint)
        else:
            c, f0, _ = track_f0(xs, sr, hop_s=0.01, hint_hz=f0_hint, resolution=0.02 if f0_hint else 0.1)
            f0 = f0[np.isfinite(f0)]
        if len(f0) < 8:
            return None
        med = np.median(f0)
        cents = 1200 * np.log2(f0 / med)
        k = max(3, int(0.4 / 0.01) | 1)
        cents_hf = cents - dsp.smooth(cents, k)
        depth = float(np.std(cents_hf))
        n = 1 << int(np.ceil(np.log2(len(cents_hf) * 4)))
        S = np.abs(np.fft.rfft(cents_hf * np.hanning(len(cents_hf)), n=n)) ** 2
        fr = np.fft.rfftfreq(n, 0.01)
        band = (fr >= 2) & (fr <= 12)
        rate = float(fr[band][np.argmax(S[band])]) if band.any() and S[band].max() > 0 else 0.0
        return dict(f0=float(med), depth=depth, rate=rate)

    so, sr_ = stats(o), stats(r)
    if so is None or sr_ is None:
        return dict(D_pitch=0.0, pitch=None)
    off = abs(1200 * np.log2(sr_["f0"] / so["f0"]))
    d_depth = abs(sr_["depth"] - so["depth"])
    d_rate = abs(sr_["rate"] - so["rate"]) if so["depth"] > 5 else 0.0
    D = off / 10.0 + d_depth / 10.0 + d_rate / 2.0
    return dict(D_pitch=float(D), pitch=dict(orig=so, rend=sr_, offset_cents=float(off)))


def evaluate_recreation(orig: np.ndarray, rend: np.ndarray, sr: int, held_range: tuple[float, float] | None = None,
                        loop_period_s: float | None = None, weights: dict | None = None,
                        loop_start_s: float | None = None, f0_hint: float | None = None,
                        track_hz: float | None = None) -> dict:
    """Score a rendered recreation against the original note. Returns a dict with score and terms."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    o, r = _align(np.asarray(orig, np.float32), np.asarray(rend, np.float32))
    if o.shape[1] != r.shape[1]:
        o, r = dsp.to_mono(o)[:, None], dsp.to_mono(r)[:, None]
    D_loud, gain = loudness_distance(o, r, sr)
    r = r * 10 ** (gain / 20)  # compare timbre after global level match
    D_spec = spectral_distance(o, r, sr)
    seam = seam_distance(o, r, sr, held_range, loop_period_s, loop_start_s)
    var = variation_distance(o, r, sr, held_range)
    pit = pitch_distance(o, r, sr, held_range, f0_hint, track_hz)
    pit["D_pitch"] = min(pit["D_pitch"], 3.0)  # pitch tracking is unreliable on inharmonic material
    z = (w["spec"] * D_spec + w["loud"] * D_loud + w["seam"] * seam["D_seam"] + w["var"] * var["D_var"]
         + w["pitch"] * pit["D_pitch"])
    out = dict(score=float(np.exp(-z)), z=float(z), D_spec=float(D_spec), D_loud=float(D_loud), gain_db=float(gain))
    out.update(seam)
    out.update(var)
    out.update(pit)
    return out


def calibrate(rows: list[dict], ratings: list[float]) -> dict:
    """Fit metric weights to human ratings in [0,1] (e.g. MUSHRA/100).

    rows: list of evaluate_recreation() outputs; ratings: matching mean opinion scores in [0, 1].
    Solves a non-negative least squares on -log(rating) = sum_i w_i D_i.
    """
    from scipy.optimize import nnls

    keys = ["D_spec", "D_loud", "D_seam", "D_var", "D_pitch"]
    A = np.array([[row[k] for k in keys] for row in rows])
    b = -np.log(np.clip(np.asarray(ratings, float), 1e-3, 1.0))
    wts, _ = nnls(A, b)
    return dict(zip(["spec", "loud", "seam", "var", "pitch"], map(float, wts)))
