"""Algorithm A: length-constrained harmonic-plus-noise loop construction.

Pipeline (see README): segment -> f0 -> harmonic analysis (+inharmonicity) ->
loop-point search in the parameter domain -> per-partial phase/amplitude
closed loop (parameter-domain crossfade + Laroche frequency nudge) ->
periodic residual noise -> envelope factorisation into coherent SFZ regions
-> WAV + SFZ -> optional sfizz render + Metric B fitness for candidate choice.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from . import dsp
from .dsp import Segmentation
from .envelope import (EnvComponent, FilEG, fit_envelope_components, fit_fileg, fit_release_time,
                       make_exact_at_start, tau_to_sfz_time)
from .harmonic import (HarmonicModel, analyze, analyze_peaks, estimate_f0_spectral, estimate_inharmonicity,
                       harmonic_centroid, interp_tracks, phase_advance, refine_octave, synth_partials, track_f0)


# ------------------------------------------------------------------ config
@dataclass
class LoopConfig:
    q: float = 0.7                    # quality / size knob in [0, 1]
    stages: int | None = None         # 1 or 2 coherent regions; None = from q
    residual: bool | None = None      # synthesize the stochastic residual; None = from q
    K_max: int | None = None          # partial count cap; None = from q
    n_candidates: int = 4             # loop candidates rendered + scored with Metric B
    verify: bool = True               # render with sfizz and score
    note_hint: str | None = None      # e.g. "a4" to constrain the f0 search
    keep_attack_max_s: float = 0.5    # latest loop start relative to attack end (scaled by q)
    seed: int = 0
    out_format: str = "wav"           # 'wav' | 'flac'
    baseline: bool = False            # also emit a classic crossfade loop for comparison
    fileg: bool = True
    hybrid: str = "auto"              # 'auto' | 'on' | 'off': keep original audio in the loop, resynthesize only
                                      # a bridge at the loop end (sustaining notes)
    f0_method: str = "spectral"       # 'spectral' (fast harmonic summation) | 'pyin' (slower, more robust on noise)
    max_total_s: float | None = None  # hard budget: total seconds of audio over all emitted files
    delay_offset_samples: int = 2     # sfizz starts a `delay`ed voice this many samples late (measured)
    stage_files: str = "delay"        # 'delay': extra stage/noise files hold only the loop and start via the
                                      # delay opcode; 'padded': zero-filled attack (player agnostic, bigger)
    method: str = "hybrid"            # 'hybrid': tracked partials / hybrid loops (default path)
                                      # 'laroche': frozen loop-locked oscillator bank + separate noise loop (sfzc.laroche)
    frozen: str = "auto"              # laroche: 'auto' (frozen unless vibrato/tremolo), 'on', 'off'
    target_periods: float = 166.0     # laroche: preferred loop length in fundamental periods (Laroche's 0.5/0.003)
    refine: bool = False              # laroche: Stage-3 MR-STFT refinement of partial / noise-band gains (PyTorch)
    lfo: bool = False                 # laroche: add gentle pitch/amp LFO opcodes to mask loop periodicity
    loop_crossfade_s: float = 0.0     # laroche: emit loop_crossfade (sfizz/OpenMPT) as a safety net
    round_robin: int = 1              # laroche: number of alternating loop sets (seq_length/seq_position)

    def resolved(self, K_all: int) -> dict:
        q = float(np.clip(self.q, 0.0, 1.0))
        stages = self.stages if self.stages is not None else (3 if q >= 0.75 else 2 if q >= 0.4 else 1)
        residual = self.residual if self.residual is not None else (q >= 0.15)
        K = self.K_max if self.K_max is not None else int(round(12 + (max(K_all, 12) - 12) * q ** 1.2))
        return dict(q=q, stages=stages, residual=residual, K=max(4, min(K, K_all if K_all > 0 else K)))


@dataclass
class LoopResult:
    name: str
    sfz_path: str
    wav_paths: list[str]
    klass: str
    key: int
    cents: float
    f0: float
    B: float
    loop_start: int
    loop_end: int
    loop_len: int
    t0_s: float
    stages: int
    residual: bool
    K: int
    duration_s: float
    original_duration_s: float
    size_bytes: int
    total_audio_s: float = 0.0
    metric: dict | None = None
    baseline_metric: dict | None = None
    candidates: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------- helpers
def _wrap(p):
    return (p + np.pi) % (2 * np.pi) - np.pi


def _detrend_log_span(tracks: np.ndarray, off: int, n_fit: int) -> np.ndarray:
    """Remove the linear log-amplitude trend fitted over samples [off, off+n_fit) of each track.

    tracks: (..., N).  The trend is removed over the whole length (relative to sample `off`) so
    that the loop becomes amplitude-stationary; the removed slope is left to the SFZ envelope.
    """
    shp = tracks.shape
    flat = tracks.reshape(-1, shp[-1])
    out = flat.copy()
    n = np.arange(shp[-1], dtype=float) - off
    nf = np.arange(n_fit, dtype=float)
    nf_c = nf - nf.mean()
    den = np.sum(nf_c ** 2) + 1e-20
    for r in range(flat.shape[0]):
        y = flat[r, off: off + n_fit]
        if y.max() <= 0 or np.count_nonzero(y > 0) < n_fit // 2:
            continue
        ly = np.log(np.maximum(y, 1e-9))
        b = np.sum(nf_c * (ly - ly.mean())) / den
        out[r] = flat[r] * np.exp(-b * n)
    return out.reshape(shp)


def _close_loop_tracks(track: np.ndarray, N: int, nX: int, X_pre: int, mode: str, off: int) -> np.ndarray:
    """Close a parameter track into an exactly repeating loop.

    ``track`` covers absolute local samples [-X_pre - nX, N) (index 0 == -X_pre - nX); the loop
    occupies [0, N).  Over the last nX samples of the loop the track is cross-faded (parameter
    domain, so no comb filtering) towards the material that naturally precedes the loop start
    (mode 'pre') or towards the loop-start value (mode 'const'), so that L(N-1) flows into L(0).
    Returns the track over [-X_pre, N): the pre-roll is the end of the closed loop, which flows
    into the unmodified loop start and hence into the original attack at the junction.
    """
    L = track[..., off: off + N].copy()
    if nX > 0:
        w = 0.5 - 0.5 * np.cos(np.pi * np.arange(nX) / max(nX - 1, 1))  # 0 -> 1
        if mode == "pre":
            P = track[..., off - nX: off]                 # material just before the loop start
            L[..., N - nX:] = (1 - w) * L[..., N - nX:] + w * P
        else:
            L[..., N - nX:] = (1 - w) * L[..., N - nX:] + w * L[..., :1]
    out = np.empty(track.shape[:-1] + (X_pre + N,))
    out[..., :X_pre] = L[..., N - X_pre: N] if X_pre <= N else np.tile(L, 2)[..., 2 * N - X_pre: 2 * N]
    out[..., X_pre:] = L
    return out


def _periodic_noise(resid_mag: np.ndarray, resid_freqs: np.ndarray, win_sq_sum: float, sr: int, N: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Exactly periodic (period N) noise whose PSD matches the residual magnitude (nbins,)."""
    # Welch PSD (one-sided, per Hz) from the analysis STFT: P = 2 |X|^2 / (sr * sum w^2)
    P = 2.0 * resid_mag.astype(np.float64) ** 2 / (sr * win_sq_sum)
    fbins = np.fft.rfftfreq(N, 1 / sr)
    Pn = np.interp(fbins, resid_freqs, P)
    # spectral line magnitude so that the power per line equals P * (sr / N)
    R = np.sqrt(N * sr * Pn / 2.0)
    ph = rng.uniform(0, 2 * np.pi, len(R))
    Y = R * np.exp(1j * ph)
    Y[0] = 0.0
    if N % 2 == 0:
        Y[-1] = Y[-1].real
    y = np.fft.irfft(Y, n=N)
    return y


def _seam_cost_search(model: HarmonicModel, i_a: int, i_r: int, N_min: int, N_max: int, i0_max: int,
                      n_ctx: int, period: float, n_top: int = 8, budget: tuple[int, int, int] | None = None
                      ) -> list[tuple[float, int, int]]:
    """Parameter-domain loop-point search (all channels).

    Loop lengths are whole numbers of fundamental periods (so harmonic partials close with
    almost no frequency nudge); starts are on the analysis frame grid.
    Returns [(cost, i0, N_samples)] sorted ascending.
    """
    C, K, F = model.freq.shape
    hop, sr = model.hop, model.sr
    body = slice(i_a, max(i_a + 1, i_r))
    fref = model.freq_mean[:, body].mean(axis=1, keepdims=True)  # (K, 1) reference for cents
    logA, cents, W, Phi = [], [], [], []
    fi = np.arange(F, dtype=float)
    bsel = np.zeros(F, bool)
    bsel[body] = True

    def detrend(tr: np.ndarray) -> np.ndarray:
        # remove a per-partial linear trend fitted over the body: the exponential decay / drift of
        # every partial is handled by the loop detrending and the envelope, so only the
        # fluctuation pattern (vibrato, tremolo, beating) has to match across the seam
        tb = fi[bsel]
        A_ = np.vstack([tb, np.ones_like(tb)]).T
        coef, *_ = np.linalg.lstsq(A_, tr[:, bsel].T, rcond=None)   # (2, K)
        return tr - (coef[0][:, None] * fi[None, :] + coef[1][:, None])

    Wt = []  # time-varying audibility weights (power fraction per frame)
    for ch in range(C):
        la = 20 * np.log10(model.amp[ch] + 1e-6)
        la = np.maximum(la, la.max(axis=0, keepdims=True) - 60.0)  # decayed partials: floor, not -120 dB
        logA.append(detrend(la))
        pf = model.amp[ch] ** 2 / ((model.amp[ch] ** 2).sum(axis=0, keepdims=True) + 1e-20)
        Wt.append(pf ** 0.75 / C)
        cents.append(detrend(1200 * np.log2(np.maximum(model.freq[ch], 1e-3) / fref)))
        pw = (model.amp[ch, :, body] ** 2).mean(axis=1)
        w = (pw / (pw.max() + 1e-20)) ** 0.75
        W.append(w / (w.sum() + 1e-20) / C)
        dphi = 2 * np.pi * hop * 0.5 * (model.freq[ch, :, 1:] + model.freq[ch, :, :-1]) / sr
        Phi.append(np.concatenate([np.zeros((K, 1)), np.cumsum(dphi, axis=1)], axis=1))
    # static-loop penalty: the track variance (vibrato / tremolo) that a loop of length N cannot
    # contain = long-term variance minus the mean within-window variance over N-long windows
    feats = []
    for ch in range(C):
        for arr, scale in ((logA[ch][:, body], 1.0), (cents[ch][:, body], 1.0 / 9.0)):
            feats.append((arr, W[ch] * scale))
    Fb = feats[0][0].shape[1]

    def static_penalty(iN: int) -> float:
        n = max(2, min(iN, Fb))
        pen = 0.0
        for arr, w in feats:
            cs = np.cumsum(np.concatenate([np.zeros((arr.shape[0], 1)), arr], axis=1), axis=1)
            cs2 = np.cumsum(np.concatenate([np.zeros((arr.shape[0], 1)), arr ** 2], axis=1), axis=1)
            mean_w = (cs[:, n:] - cs[:, :-n]) / n
            var_w = (cs2[:, n:] - cs2[:, :-n]) / n - mean_w ** 2
            var_long = arr.var(axis=1)
            pen += float(np.sum(w * np.maximum(var_long - var_w.mean(axis=1), 0.0)))
        return pen

    results = []
    m_min = max(2, int(np.ceil(N_min / period)))
    m_max = max(m_min, int(N_max // period))
    m_vals = np.unique(np.round(np.geomspace(m_min, m_max, 40)).astype(int))
    for m in m_vals:
        N = int(round(m * period))
        iN = N // hop
        frac = N - iN * hop
        hi = min(i0_max, i_r - iN - 1 - n_ctx)
        if budget is not None:
            # (loop start - onset) + n_files * N  <=  budget samples
            onset, budget_samples, n_files = budget
            latest = onset + budget_samples - n_files * N
            hi = min(hi, int(np.searchsorted(model.centers, latest, side="right")) - 1)
        if hi < i_a:
            continue
        i0s = np.arange(i_a, hi + 1)
        cost = np.zeros(len(i0s)) + static_penalty(iN)
        for ch in range(C):
            # a partial only matters at the seam if it is audible on both sides of it
            wk = np.minimum(Wt[ch][:, i0s], Wt[ch][:, i0s + iN])
            wk = wk / (wk.sum(axis=0, keepdims=True) + 1e-12) / C
            d = np.zeros(len(i0s))
            for c in range(n_ctx):
                a0, a1 = logA[ch][:, i0s + c], logA[ch][:, i0s + iN + c]
                f0, f1 = cents[ch][:, i0s + c], cents[ch][:, i0s + iN + c]
                d += (wk * ((a0 - a1) ** 2 + (f0 - f1) ** 2 / 9.0)).sum(axis=0)
            d /= n_ctx
            # cumulative phase at the exact loop end (linear within the hop)
            phi_end = Phi[ch][:, i0s + iN] + frac * 2 * np.pi * model.freq[ch][:, i0s + iN] / sr
            dphase = _wrap(phi_end - Phi[ch][:, i0s])
            shift_hz = np.abs(dphase) * sr / (2 * np.pi * N)
            nudge_cents = 1731.0 * shift_hz / np.maximum(model.freq[ch][:, i0s], 1.0)
            cost += d + (wk * nudge_cents ** 2).sum(axis=0)
        cost *= (1.3 - 0.3 * N / max(N_max, 1))  # mild preference for longer loops
        for j in range(len(i0s)):
            results.append((float(cost[j]), int(i0s[j]), int(N)))
    results.sort(key=lambda r: r[0])
    # keep the best candidate of every factor-2 length band (so the metric can arbitrate between
    # genuinely different loop lengths), then fill up with the best remaining distinct ones
    picked: list[tuple[float, int, int]] = []
    best_in_band: dict[int, tuple[float, int, int]] = {}
    for r in results:
        band = int(np.floor(np.log2(max(r[2], 1) / max(N_min, 1))))
        if band not in best_in_band:
            best_in_band[band] = r
    # longer loops first: they contain more of the natural variation; the metric arbitrates
    for band in sorted(best_in_band, reverse=True):
        picked.append(best_in_band[band])
    picked = picked[:n_top]
    for r in results:
        if len(picked) >= n_top:
            break
        if all(abs(r[1] - p[1]) >= n_ctx or abs(r[2] - p[2]) > 0.1 * p[2] for p in picked):
            picked.append(r)
    return picked


def _classic_crossfade_loop(x: np.ndarray, sr: int, seg: Segmentation, n0: int, N: int, f0: float) -> np.ndarray:
    """Baseline: crossfade loop of the same region (loop length snapped to whole periods)."""
    period = sr / f0
    N = int(round(round(N / period) * period))
    N = max(N, int(round(period)))
    nX = max(int(0.25 * N), 8)
    nX = min(nX, n0 - seg.onset)
    body = x[n0: n0 + N].copy()
    pre = x[n0 - nX: n0]
    w = dsp.raised_cosine(nX)[:, None]
    body[N - nX:] = np.sqrt(1 - w) * body[N - nX:] + np.sqrt(w) * pre  # equal-power crossfade
    out = np.concatenate([x[seg.onset: n0], body], axis=0)
    return out, N


# ---------------------------------------------------------------- pipeline
class SampleLooper:
    def __init__(self, path: str, cfg: LoopConfig | None = None):
        self.path = path
        self.cfg = cfg or LoopConfig()
        self.name = os.path.splitext(os.path.basename(path))[0].replace(" ", "_").replace("#", "s")
        self.x, self.sr = dsp.load_audio(path)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.log: list[str] = []
        self.mode = "harmonic"

    # ------------------------------------------------------------ analysis
    def analyse(self):
        x, sr = self.x, self.sr
        hint_hz = dsp.midi_to_hz(dsp.name_to_midi(self.cfg.note_hint)) if self.cfg.note_hint else None
        seg0 = dsp.segment_note(x, sr, f0_hint=hint_hz)
        if self.cfg.f0_method == "pyin":
            centers, f0, vprob = track_f0(x, sr, hint_hz=hint_hz)
            body = (centers >= seg0.attack_end) & (centers <= seg0.release_onset)
            if body.sum() < 3:
                body = (centers >= seg0.onset) & (centers <= seg0.end)
            f0b = f0[body]
            voiced = np.isfinite(f0b)
            voiced_frac = float(voiced.mean()) if voiced.size else 0.0
            f0_nom = float(np.median(f0b[voiced])) if voiced.sum() >= 3 else (hint_hz or 0.0)
            pitched = voiced_frac > 0.4 and f0_nom > 0
        else:
            f0_nom, sal = estimate_f0_spectral(x, sr, seg0.attack_end, max(seg0.release_onset, seg0.attack_end + sr // 10),
                                               hint_hz=hint_hz)
            voiced_frac = sal / 6.0
            pitched = sal > 2.0 and f0_nom > 0
            centers, f0 = np.array([0, len(x)]), np.array([f0_nom, f0_nom])
        if pitched and hint_hz is None:
            f0_nom = refine_octave(x, sr, f0_nom, seg0.attack_end, seg0.release_onset)
        budget = self.cfg.max_total_s
        over_budget = budget is not None and (seg0.end - seg0.onset) / sr > budget
        seg = dsp.segment_note(x, sr, f0_hint=f0_nom if pitched else None, force_loop=bool(over_budget and pitched))
        self.seg, self.f0_track, self.f0_nom, self.pitched, self.voiced_frac = seg, (centers, f0), f0_nom, pitched, voiced_frac
        self.log.append(f"class={seg.klass} pitched={pitched} f0={f0_nom:.2f}Hz voiced={voiced_frac:.2f} "
                        f"onset={seg.onset/sr:.3f}s attack_end={seg.attack_end/sr:.3f}s "
                        f"release_onset={seg.release_onset/sr:.3f}s end={seg.end/sr:.3f}s slope={seg.body_slope_db_s:.1f}dB/s")
        if not pitched or seg.klass == "oneshot":
            self.model = None
            return
        K_all = int(np.floor((sr / 2 - 200) / f0_nom))
        self.res = self.cfg.resolved(K_all)
        # inharmonicity only matters for struck/plucked (decaying) notes; sustaining notes skip the pre-pass
        prepass = seg.klass == "decay"
        model = analyze(x, sr, self.f0_track, f0_nom, K_max=min(40, K_all) if prepass else self.res["K"],
                        n_start=seg.onset, n_end=seg.end, residual=not prepass)
        i_a, i_r = model.frame_at(seg.attack_end), model.frame_at(seg.release_onset)
        # a harmonic model needs a dense low harmonic series; bells / drums / spurious sub-harmonic
        # f0 estimates give a sparse one -> treat as unpitched (one-shot)
        body = slice(i_a, max(i_a + 1, i_r + 1))
        a_b = model.amp_mean[:, body]
        hidx = np.array(model.extra.get("harmonic_index", list(range(1, model.K + 1))))
        strong = a_b > a_b.max(axis=0, keepdims=True) * 10 ** (-35 / 20)
        dense = float(np.median(strong[hidx <= 12].sum(axis=0))) if a_b.size else 0.0
        dense_lo = float(np.median(strong[hidx <= 6].sum(axis=0))) if a_b.size else 0.0
        cen = float(harmonic_centroid(a_b.mean(axis=1), model.freq_mean[:, body].mean(axis=1)))
        self.log.append(f"low-harmonic density: {dense:.0f} of {min(12, model.K)} (k<=6: {dense_lo:.0f}) strong, "
                        f"centroid/f0={cen / f0_nom:.1f}")
        if dense < 5 or dense_lo < 3 or cen / f0_nom > 30:
            # sparse spectrum: try an inharmonic peak-tracking model (bells, mallets, plucked metal)
            pm = analyze_peaks(x, sr, seg.onset, seg.end, (seg.attack_end, seg.release_onset), K_max=self.res["K"])
            if pm is not None:
                ia2, ir2 = pm.frame_at(seg.attack_end), pm.frame_at(seg.release_onset)
                hp = float(np.median(pm.harmonicity[ia2: ir2 + 1])) if ir2 > ia2 else 0.0
                self.log.append(f"peak mode: {pm.K} peaks {np.round(pm.extra['peak_freqs'][:6], 1).tolist()}... "
                                f"sinusoidal energy fraction {hp:.2f}")
                if hp >= 0.3 and pm.K >= 2:
                    self.model = pm
                    self.f0_nom = pm.f0_nominal
                    self.mode = "peaks"
                    return
            self.pitched = False
            self.model = None
            return
        B = 0.0
        if prepass:
            B = estimate_inharmonicity(model, i_a, i_r)
            if B < 2e-6:
                B = 0.0
            model = analyze(x, sr, self.f0_track, f0_nom, K_max=self.res["K"], B=B, n_start=seg.onset, n_end=seg.end)
        self.model = model
        harm = float(np.median(model.harmonicity[i_a:i_r + 1])) if i_r > i_a else float(np.median(model.harmonicity))
        self.log.append(f"K={model.K} M={model.M} hop={model.hop} B={B:.2e} harmonicity={harm:.2f}")
        if harm < 0.05:
            # essentially unpitched -> treat as one-shot
            self.pitched = False
            self.model = None

    # ------------------------------------------------------------- one-shot
    def _write_oneshot(self, out_dir: str) -> LoopResult:
        x, sr, seg = self.x, self.sr, self.seg
        y = x[seg.onset: seg.end + 1].copy()
        if self.cfg.max_total_s is not None and len(y) > int(self.cfg.max_total_s * sr):
            y = y[: int(self.cfg.max_total_s * sr)]
            nfo = min(len(y), int(0.02 * sr))
            y[-nfo:] *= dsp.raised_cosine(nfo)[::-1, None]
            self.log.append(f"one-shot truncated to the {self.cfg.max_total_s:.2f}s budget with a 20 ms fade")
        nf = min(len(y), int(0.002 * sr))
        y[:nf] *= dsp.raised_cosine(nf)[:, None]
        y[-nf:] *= dsp.raised_cosine(nf)[::-1, None]
        wav = os.path.join(out_dir, f"{self.name}.{self.cfg.out_format}")
        dsp.write_audio(wav, y, sr, subtype="PCM_16")
        key = int(round(dsp.hz_to_midi(self.f0_nom))) if self.pitched else 60
        cents = 100 * (dsp.hz_to_midi(self.f0_nom) - key) if self.pitched else 0.0
        rel = fit_release_time(seg.env_t, seg.env_db, sr, seg.release_onset, seg.end, default=0.3)
        mode = "one_shot" if not self.pitched else "no_loop"
        sfz = os.path.join(out_dir, f"{self.name}.sfz")
        with open(sfz, "w") as f:
            f.write(f"// sfzc one-shot ({'unpitched' if not self.pitched else 'too short to loop'})\n")
            f.write(f"<region> sample={os.path.basename(wav)} pitch_keycenter={key} lokey={key} hikey={key} "
                    f"loop_mode={mode} amp_veltrack=0 ampeg_release={rel:.3f}\n")
        return LoopResult(self.name, sfz, [wav], "oneshot", key, float(cents), self.f0_nom, 0.0, 0, 0, 0, 0.0, 1,
                          False, 0, len(y) / sr, len(x) / sr, os.path.getsize(wav), total_audio_s=len(y) / sr,
                          info={"log": self.log})

    # ------------------------------------------------------------ building
    def _candidates(self, n_files: int | None = None):
        model, seg, sr = self.model, self.seg, self.sr
        q = self.res["q"]
        i_a, i_r = model.frame_at(seg.attack_end), model.frame_at(seg.release_onset)
        hop = model.hop
        period = sr / self.f0_nom
        budget = None
        if self.cfg.max_total_s is not None:
            # under a hard budget the loop may start right after the onset transient
            hard_attack = min(seg.attack_end, seg.onset + int(0.1 * sr) + int(2 * period))
            i_a = model.frame_at(max(hard_attack, seg.onset + int(0.02 * sr)))
            budget = (seg.onset, int(self.cfg.max_total_s * sr), n_files or 1)
        L_full = max(0, (i_r - i_a) * hop)
        N_min = int(max(2 * period, 0.005 * sr))
        N_max_q = int(N_min + (0.8 * L_full - N_min) * q ** 1.5)
        # the budget is in samples on disk: the parametric path writes one file per envelope stage
        # (+ one for the residual), the hybrid path a single file
        hybrid_possible = self.cfg.hybrid != "off" and seg.klass == "sustain"
        if n_files is None:
            n_files = 1 if hybrid_possible else (self.res["stages"] + (1 if self.res["residual"] else 0))
        N_max = max(N_min, int(N_max_q / n_files))
        if budget is not None:
            N_max = max(N_min, min(N_max, (budget[1] - (int(model.centers[i_a]) - seg.onset)) // n_files))
        i0_max = i_a + int(self.cfg.keep_attack_max_s * q * sr / hop)
        n_ctx = max(2, int(round(0.04 * sr / hop)))
        cands = _seam_cost_search(model, i_a, i_r, N_min, N_max, i0_max, n_ctx, period,
                                  n_top=max(8, self.cfg.n_candidates), budget=budget)
        if not cands:
            # degenerate: body too short -> shortest loop at attack end
            cands = [(1e9, i_a, int(round(max(2, np.ceil(N_min / period)) * period)))]
        self.log.append(f"loop search ({n_files} file{'s' if n_files != 1 else ''}): L_full={L_full/sr:.3f}s "
                        f"N_min={N_min/sr:.4f}s N_max={N_max/sr:.3f}s -> {len(cands)} candidates, best cost={cands[0][0]:.2f}")
        return cands

    def build(self, i0: int, N: int, stages: int, residual: bool, out_dir: str, tag: str = "") -> dict:
        model, seg, sr, x = self.model, self.seg, self.sr, self.x
        C = x.shape[1]
        hop = model.hop
        n0 = int(model.centers[i0])
        N = int(N)
        iN = N // hop
        X_pre = int(min(model.M // 2, n0 - seg.onset, max(int(0.002 * sr), 1)))
        X_pre = max(X_pre, 1)
        # loop-closing blend: towards the material preceding the loop start when it is sustain-like
        nX = max(1, int(0.25 * N))
        pre_avail = n0 - X_pre - seg.attack_end
        if pre_avail >= max(int(0.01 * sr), 1):
            mode, nX = "pre", int(min(nX, pre_avail))
        else:
            mode = "const"
        nX_eff = nX
        span = X_pre + (nX if mode == "pre" else 0)   # index of the loop start in the gathered tracks
        lo = n0 - span
        a_raw = interp_tracks(model.centers, model.amp, lo, n0 + N)      # (C, K, span + N)
        f_raw = interp_tracks(model.centers, model.freq, lo, n0 + N)     # (C, K, span + N)
        # remove the slow amplitude trend over the loop (handled by the SFZ envelope)
        a_det = _detrend_log_span(a_raw, span, N)
        A = _close_loop_tracks(a_det, N, nX, X_pre, mode, span)
        Fq = _close_loop_tracks(f_raw, N, nX, X_pre, mode, span)
        # micro-variation pattern normalised to mean 1 over the loop
        mean_loop = A[..., X_pre:].mean(axis=-1, keepdims=True)
        m = np.where(mean_loop > 1e-9, A / np.maximum(mean_loop, 1e-12), 1.0)
        # loop-closing frequency nudge (Laroche), per channel and partial
        adv = phase_advance(Fq[..., X_pre:], sr)                # (C, K)
        slope = -_wrap(adv) / N
        nudge_cents = 1731.0 * np.abs(slope) * sr / (2 * np.pi) / np.maximum(model.freq[:, :, i0], 1.0)
        wn = model.amp[:, :, i0] ** 2
        nudge_w = float((nudge_cents * wn).sum() / (wn.sum() + 1e-20))
        # envelope factorisation over the body
        i_r = model.frame_at(seg.release_onset)
        body = slice(i0, max(i0 + 2, i_r + 1))
        t_body = (model.centers[body] - n0) / sr
        amp_body = model.amp[:, :, body]
        klass = seg.klass
        comps, fit_err, base_err = fit_envelope_components(amp_body, t_body, klass, stages)
        amp0 = mean_loop[..., 0] * 1.0  # (C, K) loop level after detrending == level at t0
        make_exact_at_start(comps, amp0)
        # synthesise each stage, phase anchored to the analysed phase at the loop start
        phase0 = model.phase[:, :, i0]
        ys = []
        for comp in comps:
            amp_s = comp.coef[:, :, None] * m
            ys.append(synth_partials(amp_s, Fq, phase0, sr, slope=slope, phase_ref=X_pre))
        # residual noise (stage 1 only), exactly periodic with period N
        noise_rms = 0.0
        if residual:
            rc = model.resid_centers
            fr = np.where((rc >= n0 - model.M) & (rc <= n0 + N + model.M))[0]
            if fr.size == 0:
                fr = np.array([int(np.argmin(np.abs(rc - n0)))])
            nz = np.zeros((X_pre + N, C))
            g_frames = np.sqrt((model.resid_mag[0][:, fr].astype(np.float64) ** 2).sum(axis=0) + 1e-20)
            if fr.size >= 2:
                g = np.interp(np.arange(lo, n0 + N), rc[fr], g_frames)
            else:
                g = np.full(n0 + N - lo, g_frames[0])
            g = _detrend_log_span(g[None, None, :], span, N)[0, 0]
            g = _close_loop_tracks(g, N, nX_eff, X_pre, mode, span)
            g = g / (g[X_pre:].mean() + 1e-20)
            for q_i in range(2 if C == 2 else 1):
                mag = np.sqrt((model.resid_mag[q_i][:, fr].astype(np.float64) ** 2).mean(axis=1))
                per = _periodic_noise(mag, model.resid_freqs, model.resid_win_sq_sum, sr, N, self.rng)
                idx = (np.arange(-X_pre, N)) % N
                comp_sig = per[idx]
                if C == 2:
                    nz[:, 0] += comp_sig
                    nz[:, 1] += comp_sig if q_i == 0 else -comp_sig
                else:
                    nz[:, 0] += comp_sig
            nz *= g[:, None]
            noise_rms = float(np.sqrt(np.mean(nz[X_pre:] ** 2)))
            # the residual gets its own region(s): its level follows the recorded residual level
            body_r = np.where((rc >= n0) & (rc <= seg.release_onset))[0]
            if body_r.size >= 3:
                gl = np.sqrt((model.resid_mag[0][:, body_r].astype(np.float64) ** 2).sum(axis=0) + 1e-20)
                gl = gl / (gl[:max(1, min(3, len(gl)))].mean() + 1e-20)
                ncomps, _, _ = fit_envelope_components(gl[None, None, :], (rc[body_r] - n0) / sr, klass, min(stages, 2))
                tot = sum(float(c.coef[0, 0]) for c in ncomps)
                noise_comps = [(c, float(c.coef[0, 0]) / tot) for c in ncomps if tot > 0 and c.coef[0, 0] / tot > 0.02]
            else:
                noise_comps = [(EnvComponent("const"), 1.0)]
        # assemble WAVs. File 0 = original attack + cross-fade + loop. Extra stage / noise files hold
        # only the pre-roll + loop and are started with the `delay` opcode (fade-in via ampeg_attack),
        # unless stage_files == 'padded' (zero-filled attack, no reliance on delay accuracy).
        w_in = dsp.raised_cosine(X_pre)[:, None]
        pre_len = n0 - X_pre - seg.onset
        padded = self.cfg.stage_files == "padded"
        wavs = []
        for j, y in enumerate(ys):
            if j == 0:
                head = x[seg.onset: n0 - X_pre].copy()
                xf = x[n0 - X_pre: n0] * (1 - w_in) + y[:X_pre] * w_in
                out = np.concatenate([head, xf, y[X_pre:]], axis=0)
                nf = min(len(out), int(0.002 * sr))
                if nf > 1:
                    out[:nf] *= dsp.raised_cosine(nf)[:, None]
            elif padded:
                out = np.concatenate([np.zeros((pre_len, C)), y[:X_pre] * w_in, y[X_pre:]], axis=0)
            else:
                out = y.copy()  # pre-roll (X_pre) + loop; the player fades it in
            wavs.append(out)
        if residual:
            if padded:
                wavs.append(np.concatenate([np.zeros((pre_len, C)), nz[:X_pre] * w_in, nz[X_pre:]], axis=0))
            else:
                wavs.append(nz.copy())
        loop_start = n0 - seg.onset
        loop_end = loop_start + N - 1
        t_hold = loop_start / sr
        rel = fit_release_time(seg.env_t, seg.env_db, sr, seg.release_onset, seg.end,
                               default=0.4 if klass == "sustain" else 0.25)
        regions = []
        # extra files start X_pre samples before the loop start; compensate sfizz's fixed voice-start
        # latency for `delay`ed regions so that all stages sum sample-coherently
        delay_s = (pre_len - self.cfg.delay_offset_samples + 0.5) / sr
        extra_pos = "" if padded else (f"delay={delay_s:.6f} ampeg_attack={X_pre / sr:.6f} "
                                       f"loop_start={X_pre} loop_end={X_pre + N - 1}")

        def env_opcodes(comp, extra: bool = False):
            hold = (X_pre / sr) if (extra and not padded) else t_hold
            if comp.kind == "const":
                return "ampeg_sustain=100"
            return f"ampeg_hold={hold:.5f} ampeg_decay={tau_to_sfz_time(comp.tau):.4f} ampeg_sustain=0"

        for j, (comp, y) in enumerate(zip(comps, wavs)):
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            gain_db = 0.0
            if peak > 0.98:
                y *= 0.98 / peak
                gain_db = 20 * np.log10(peak / 0.98)
            regions.append(dict(kind=comp.kind, tau=comp.tau, env=env_opcodes(comp, extra=j > 0), gain_db=gain_db,
                                wav=y, file=j, extra=j > 0))
        if residual:
            y = wavs[-1]
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            gain_db = 0.0
            if peak > 0.98:
                y *= 0.98 / peak
                gain_db = 20 * np.log10(peak / 0.98)
            for comp, frac in noise_comps:
                regions.append(dict(kind="noise-" + comp.kind, tau=comp.tau, env=env_opcodes(comp, extra=True),
                                    gain_db=gain_db + 20 * np.log10(max(frac, 1e-4)), wav=y, file=len(wavs) - 1,
                                    extra=True))
        fileg = None
        if self.cfg.fileg and klass == "decay" and stages == 1:
            tc = harmonic_centroid(amp_body.mean(axis=0), model.freq_mean[:, body])
            fileg = fit_fileg(np.abs(comps[0].coef.mean(axis=0)), model.freq_mean[:, i0], tc, t_body)
        key = int(round(dsp.hz_to_midi(self.f0_nom)))
        cents = 100 * (dsp.hz_to_midi(self.f0_nom) - key)
        wav_paths = []
        for j, y in enumerate(wavs):
            is_noise = residual and j == len(wavs) - 1
            suffix = "" if j == 0 else ("_noise" if is_noise else f"_s{j + 1}")
            p = os.path.join(out_dir, f"{self.name}{tag}{suffix}.{self.cfg.out_format}")
            dsp.write_audio(p, y, sr, subtype="PCM_16")
            wav_paths.append(p)
        sfz_path = os.path.join(out_dir, f"{self.name}{tag}.sfz")
        with open(sfz_path, "w") as f:
            f.write(f"// sfzc harmonic+noise loop | q={self.res['q']:.2f} class={klass} f0={self.f0_nom:.2f}Hz "
                    f"({cents:+.1f} cents from key {key}) B={model.B:.2e}\n")
            f.write(f"// loop {loop_start}..{loop_end} ({N} samples = {N/sr*1000:.1f} ms, {N*self.f0_nom/sr:.1f} periods), "
                    f"K={model.K}, residual={'on' if residual else 'off'}, blend={mode}/{nX_eff/sr*1000:.0f}ms, "
                    f"max partial nudge={nudge_cents.max():.2f} cents\n")
            f.write(f"// envelope fit error {fit_err:.2f} dB (single-stage baseline {base_err:.2f} dB)\n")
            f.write("<global> amp_veltrack=0\n")
            for j, r in enumerate(regions):
                pos = extra_pos if (r.get("extra") and extra_pos) else f"loop_start={loop_start} loop_end={loop_end}"
                f.write(f"<region> sample={os.path.basename(wav_paths[r['file']])} pitch_keycenter={key} lokey={key} hikey={key}"
                        f"  // {r['kind']}\n"
                        f"  loop_mode=loop_continuous {pos}\n"
                        f"  volume={r['gain_db']:.3f} {r['env']} ampeg_release={rel:.4f}\n")
                if j == 0 and fileg is not None:
                    f.write(f"  {fileg.opcodes(t_hold)}\n")
        size = sum(os.path.getsize(p) for p in wav_paths)
        total_s = sum(len(w) for w in wavs) / sr
        return dict(sfz_path=sfz_path, wav_paths=wav_paths, key=key, cents=cents, loop_start=loop_start,
                    loop_end=loop_end, N=N, n0=n0, i0=i0, iN=iN, stages=len(comps), residual=residual, total_s=total_s,
                    nudge_cents_max=float(nudge_cents.max()), nudge_cents_w=nudge_w, fit_err=fit_err, base_err=base_err,
                    fileg=asdict(fileg) if fileg else None, noise_rms=noise_rms, size=size,
                    duration_s=len(wavs[0]) / sr, release=rel, taus=[c.tau for c in comps], blend=mode)

    def _use_hybrid(self, N: int) -> bool:
        if self.cfg.hybrid == "off" or self.seg.klass != "sustain":
            return False
        return N >= 6 * self.sr / self.f0_nom

    def build_hybrid(self, i0: int, N: int, stages: int, residual: bool, out_dir: str, tag: str = "") -> dict:
        """Hybrid loop: original audio for the first ~75 % of the loop, then a resynthesised bridge
        whose partials are phase-closed onto the analysed phases at the loop start.  One WAV,
        1-2 regions (broadband constant + exponential envelope on the same file)."""
        model, seg, sr, x = self.model, self.seg, self.sr, self.x
        C = x.shape[1]
        n0 = int(model.centers[i0])
        N = int(N)
        frac = float(np.clip(0.12 * sr / max(N, 1), 0.3, 0.7))  # short loops get a proportionally longer bridge
        nB = max(int(frac * N), int(0.02 * sr))
        nB = min(nB, N - 1)
        Xb = int(min(model.M // 2, max(int(0.002 * sr), 1), N - nB - 1))
        Xb = max(Xb, 1)
        nS = nB + Xb                          # synthesised samples at the end of the loop
        s0 = n0 + N - nS                      # absolute start of the synthesis
        ib = model.frame_at(s0)
        # broadband detrend gain for the original part (slope of the total level over the loop)
        lvl = np.sqrt((model.amp_mean ** 2).sum(axis=0) + 1e-20)
        sel = (model.centers >= n0) & (model.centers < n0 + N)
        tt = (model.centers[sel] - n0) / sr
        if sel.sum() >= 3:
            A_ = np.vstack([tt, np.ones_like(tt)]).T
            b_s, _ = np.linalg.lstsq(A_, np.log(lvl[sel] + 1e-12), rcond=None)[0]
        else:
            b_s = 0.0
        gain = np.exp(-b_s * np.arange(N) / sr)
        # tracks for the bridge: [s0, n0+N) detrended with the same slope, closed towards the
        # material before the loop start (mode 'pre') or the loop-start value
        nX = nB
        pre_avail = n0 - seg.attack_end
        mode = "pre" if pre_avail >= max(int(0.01 * sr), 1) else "const"
        nX = int(min(nX, pre_avail)) if mode == "pre" else nX
        lo = n0 - (nX if mode == "pre" else 0)
        a_raw = interp_tracks(model.centers, model.amp, lo, n0 + N)
        f_raw = interp_tracks(model.centers, model.freq, lo, n0 + N)
        off = n0 - lo
        a_det = a_raw * np.exp(-b_s * (np.arange(a_raw.shape[-1]) - off) / sr)
        # close the loop over the last nX samples (parameter domain)
        A_full = _close_loop_tracks(a_det, N, nX, 1, mode, off)[..., 1:]     # (C, K, N) loop-domain
        F_full = _close_loop_tracks(f_raw, N, nX, 1, mode, off)[..., 1:]
        A_b, F_b = A_full[..., N - nS:], F_full[..., N - nS:]
        # phase closing: from the analysed phase at s0 (frame ib) to the analysed phase at n0
        phase_b = model.phase[:, :, ib]
        # phase of a free run over the bridge, then the slope that lands on phase[i0] at sample N
        adv = np.zeros((C, model.K))
        for c in range(C):
            ph = np.cumsum(2 * np.pi * F_b[c] / sr, axis=1)
            adv[c] = ph[:, -1]  # phase advance from index 0 to index nS (one past the end)
        # analysed phases are at frame centres; correct for the offset between s0 and centre(ib)
        dn = s0 - int(model.centers[ib])
        phase_at_s0 = phase_b + 2 * np.pi * model.freq[:, :, ib] * dn / sr
        target = model.phase[:, :, i0]
        slope = _wrap(target - (phase_at_s0 + adv)) / nS
        nudge_cents = 1731.0 * np.abs(slope) * sr / (2 * np.pi) / np.maximum(model.freq[:, :, i0], 1.0)
        wn = model.amp[:, :, i0] ** 2
        nudge_w = float((nudge_cents * wn).sum() / (wn.sum() + 1e-20))
        yb = synth_partials(A_b, F_b, phase_at_s0, sr, slope=slope, phase_ref=0)
        # residual noise for the bridge (periodic-N noise, natural level)
        noise_rms = 0.0
        if residual:
            rc = model.resid_centers
            fr = np.where((rc >= n0 - model.M) & (rc <= n0 + N + model.M))[0]
            if fr.size == 0:
                fr = np.array([int(np.argmin(np.abs(rc - n0)))])
            nz = np.zeros((nS, C))
            idx = np.arange(N - nS, N)
            for q_i in range(2 if C == 2 else 1):
                mag = np.sqrt((model.resid_mag[q_i][:, fr].astype(np.float64) ** 2).mean(axis=1))
                per = _periodic_noise(mag, model.resid_freqs, model.resid_win_sq_sum, sr, N, self.rng)
                comp_sig = per[idx]
                if C == 2:
                    nz[:, 0] += comp_sig
                    nz[:, 1] += comp_sig if q_i == 0 else -comp_sig
                else:
                    nz[:, 0] += comp_sig
            noise_rms = float(np.sqrt(np.mean(nz ** 2)))
            yb = yb + nz
        # assemble: original (detrended) up to s0 + crossfade over Xb into the bridge
        orig_loop = x[n0: n0 + N] * gain[:, None]
        w_in = dsp.raised_cosine(Xb)[:, None]
        loop = orig_loop.copy()
        loop[N - nS: N - nS + Xb] = orig_loop[N - nS: N - nS + Xb] * (1 - w_in) + yb[:Xb] * w_in
        loop[N - nS + Xb:] = yb[Xb:]
        out = np.concatenate([x[seg.onset: n0], loop], axis=0)
        nf = min(len(out), int(0.002 * sr))
        if nf > 1:
            out[:nf] *= dsp.raised_cosine(nf)[:, None]
        loop_start = n0 - seg.onset
        loop_end = loop_start + N - 1
        t_hold = loop_start / sr
        rel = fit_release_time(seg.env_t, seg.env_db, sr, seg.release_onset, seg.end, default=0.4)
        # broadband envelope: constant + exponential on the same file
        i_r = model.frame_at(seg.release_onset)
        body = slice(i0, max(i0 + 2, i_r + 1))
        t_body = (model.centers[body] - n0) / sr
        comps, fit_err, base_err = fit_envelope_components(lvl[body][None, None, :], t_body, "sustain", stages)
        tot = sum(float(c.coef[0, 0]) for c in comps)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        gdb = 0.0
        if peak > 0.98:
            out *= 0.98 / peak
            gdb = 20 * np.log10(peak / 0.98)
        regions = []
        for c in comps:
            frac = float(c.coef[0, 0]) / tot if tot > 0 else 1.0
            if frac <= 0.02:
                continue
            env = "ampeg_sustain=100" if c.kind == "const" else \
                f"ampeg_hold={t_hold:.5f} ampeg_decay={tau_to_sfz_time(c.tau):.4f} ampeg_sustain=0"
            regions.append(dict(kind="hybrid-" + c.kind, env=env, gain_db=gdb + 20 * np.log10(frac)))
        key = int(round(dsp.hz_to_midi(self.f0_nom)))
        cents = 100 * (dsp.hz_to_midi(self.f0_nom) - key)
        p = os.path.join(out_dir, f"{self.name}{tag}.{self.cfg.out_format}")
        dsp.write_audio(p, out, sr, subtype="PCM_16")
        sfz_path = os.path.join(out_dir, f"{self.name}{tag}.sfz")
        with open(sfz_path, "w") as f:
            f.write(f"// sfzc hybrid loop (original audio + resynthesised phase-closing bridge) | q={self.res['q']:.2f} "
                    f"class=sustain f0={self.f0_nom:.2f}Hz ({cents:+.1f} cents from key {key}) B={model.B:.2e}\n")
            f.write(f"// loop {loop_start}..{loop_end} ({N} samples = {N/sr*1000:.1f} ms, {N*self.f0_nom/sr:.1f} periods), "
                    f"bridge={nS/sr*1000:.0f}ms, K={model.K}, residual={'on' if residual else 'off'}, "
                    f"max partial nudge={nudge_cents.max():.2f} cents\n")
            f.write("<global> amp_veltrack=0\n")
            for r in regions:
                f.write(f"<region> sample={os.path.basename(p)} pitch_keycenter={key} lokey={key} hikey={key}  // {r['kind']}\n"
                        f"  loop_mode=loop_continuous loop_start={loop_start} loop_end={loop_end}\n"
                        f"  volume={r['gain_db']:.3f} {r['env']} ampeg_release={rel:.4f}\n")
        return dict(sfz_path=sfz_path, wav_paths=[p], key=key, cents=cents, loop_start=loop_start, loop_end=loop_end,
                    N=N, n0=n0, i0=i0, iN=N // model.hop, stages=len(regions), residual=residual, total_s=len(out) / sr,
                    nudge_cents_max=float(nudge_cents.max()), nudge_cents_w=nudge_w, fit_err=fit_err, base_err=base_err,
                    fileg=None,
                    noise_rms=noise_rms, size=os.path.getsize(p), duration_s=len(out) / sr, release=rel,
                    taus=[c.tau for c in comps], blend="hybrid")

    def build_any(self, i0: int, N: int, stages: int, residual: bool, out_dir: str, tag: str = "") -> dict:
        if self._use_hybrid(N):
            return self.build_hybrid(i0, N, stages, residual, out_dir, tag)
        return self.build(i0, N, stages, residual, out_dir, tag)

    def build_baseline(self, i0: int, N: int, out_dir: str) -> dict:
        model, seg, sr, x = self.model, self.seg, self.sr, self.x
        n0 = int(model.centers[i0])
        y, N = _classic_crossfade_loop(x, sr, seg, n0, N, self.f0_nom)
        p = os.path.join(out_dir, f"{self.name}_xfade.{self.cfg.out_format}")
        dsp.write_audio(p, y, sr)
        key = int(round(dsp.hz_to_midi(self.f0_nom)))
        loop_start = n0 - seg.onset
        rel = fit_release_time(seg.env_t, seg.env_db, sr, seg.release_onset, seg.end)
        sfz_path = os.path.join(out_dir, f"{self.name}_xfade.sfz")
        env = "ampeg_sustain=100"
        if seg.klass == "decay":
            comps, *_ = fit_envelope_components(model.amp[:, :, i0: model.frame_at(seg.release_onset) + 1],
                                                (model.centers[i0: model.frame_at(seg.release_onset) + 1] - n0) / sr,
                                                "decay", 1)
            env = f"ampeg_hold={loop_start/sr:.5f} ampeg_decay={tau_to_sfz_time(comps[0].tau):.4f} ampeg_sustain=0"
        with open(sfz_path, "w") as f:
            f.write("// baseline: classic equal-power crossfade loop\n<global> amp_veltrack=0\n")
            f.write(f"<region> sample={os.path.basename(p)} pitch_keycenter={key} lokey={key} hikey={key} "
                    f"loop_mode=loop_continuous loop_start={loop_start} loop_end={loop_start + N - 1} {env} "
                    f"ampeg_release={rel:.4f}\n")
        return dict(sfz_path=sfz_path, wav_paths=[p], key=key, loop_start=loop_start, N=N, size=os.path.getsize(p))

    # ------------------------------------------------------------ evaluate
    def render_protocol(self):
        """(note_on_s, render_s, original_for_comparison, held_range_s)."""
        seg, sr = self.seg, self.sr
        total = (seg.end - seg.onset) / sr
        if seg.klass == "decay":
            note_on = max(0.05, total - 0.05)
        else:
            note_on = max(0.05, (seg.release_onset - seg.onset) / sr)
        orig = self.x[seg.onset: seg.end + 1]
        held = ((seg.attack_end - seg.onset) / sr, (seg.release_onset - seg.onset) / sr)
        return note_on, total + 0.05, orig, held

    def evaluate(self, sfz_path: str, key: int, loop_len_s: float | None, loop_start_s: float | None = None) -> dict:
        from .metric import evaluate_recreation
        from .render import render_sfz

        note_on, render_s, orig, held = self.render_protocol()
        y = render_sfz(sfz_path, key, note_on, render_s, sr=self.sr)
        # vibrato/pitch statistics are measured on the strongest of the first partials
        track_hz = self.f0_nom
        if self.model is not None:
            i_a, i_r = self.model.frame_at(self.seg.attack_end), self.model.frame_at(self.seg.release_onset)
            a_b = self.model.amp_mean[:, i_a: max(i_a + 1, i_r + 1)].mean(axis=1)
            kk = int(np.argmax(a_b[: min(4, len(a_b))]))
            track_hz = float(self.model.freq_mean[kk, i_a: max(i_a + 1, i_r + 1)].mean())
        return evaluate_recreation(orig, y, self.sr, held_range=held, loop_period_s=loop_len_s,
                                   loop_start_s=loop_start_s, f0_hint=self.f0_nom, track_hz=track_hz)

    # --------------------------------------------------------------- driver
    def run(self, out_dir: str) -> LoopResult:
        os.makedirs(out_dir, exist_ok=True)
        self.analyse()
        if self.model is None:
            return self._write_oneshot(out_dir)
        res = self.res
        hybrid_possible = self.cfg.hybrid != "off" and self.seg.klass == "sustain"
        # configurations (stages, residual) to try; under a hard budget several are worth comparing
        if hybrid_possible and self.cfg.max_total_s is not None and self.cfg.verify:
            configs = [("hybrid", res["stages"], res["residual"]), ("param", 1, True), ("param", 2, True)]
        elif hybrid_possible:
            configs = [("hybrid", res["stages"], res["residual"])]
        elif self.cfg.max_total_s is not None and self.cfg.verify:
            configs = [("param", 1, False), ("param", 1, True), ("param", 2, True), ("param", 3, True)]
            configs = [(p, s, r) for p, s, r in configs if s <= max(res["stages"], 1) and (r <= res["residual"] or not r)]
        else:
            configs = [("param", res["stages"], res["residual"])]
        n_try = max(1, self.cfg.n_candidates) if self.cfg.verify else 1
        if len(configs) > 1:
            n_try = max(1, min(n_try, 2))
        tried = []
        best = None
        ci = 0
        for path_c, stages_c, resid_c in configs:
            use_hybrid = path_c == "hybrid"
            n_files = 1 if use_hybrid else stages_c + (1 if resid_c else 0)
            cands = self._candidates(n_files=n_files)
            for (cost, i0, N) in cands[:n_try]:
                tag = "" if ci == 0 else f"_c{ci}"
                ci += 1
                if use_hybrid and self._use_hybrid(N):
                    b = self.build_hybrid(i0, N, stages_c, resid_c, out_dir, tag=tag)
                else:
                    b = self.build(i0, N, stages_c, resid_c, out_dir, tag=tag)
                b["search_cost"] = cost
                b["config"] = (path_c, stages_c, resid_c)
                if self.cfg.verify:
                    b["metric"] = self.evaluate(b["sfz_path"], b["key"], b["N"] / self.sr, b["loop_start"] / self.sr)
                    score = b["metric"]["score"]
                else:
                    b["metric"] = None
                    score = -cost
                tried.append(b)
                self.log.append(f"candidate {ci - 1} ({path_c} stages={stages_c} residual={resid_c}): i0={i0} N={b['N']} "
                                f"({b['N']/self.sr*1000:.1f} ms) cost={cost:.2f} score={score:.3f} "
                                f"nudge={b['nudge_cents_w']:.1f}c (max {b['nudge_cents_max']:.0f}c) total={b['total_s']:.3f}s")
                if best is None or score > best[0]:
                    best = (score, b)
        # promote the winner to the canonical file names
        _, b = best
        if b["sfz_path"] != os.path.join(out_dir, f"{self.name}.sfz"):
            if b["config"][0] == "hybrid" and self._use_hybrid(b["N"]):
                canon = self.build_hybrid(b["i0"], b["N"], b["config"][1], b["config"][2], out_dir, tag="")
            else:
                canon = self.build(b["i0"], b["N"], b["config"][1], b["config"][2], out_dir, tag="")
            canon["metric"], canon["search_cost"] = b["metric"], b["search_cost"]
            b = canon
        for t in tried:  # remove the losing candidates' files
            if t["sfz_path"] != b["sfz_path"] and t["sfz_path"] != os.path.join(out_dir, f"{self.name}.sfz"):
                for p in t["wav_paths"] + [t["sfz_path"]]:
                    if os.path.exists(p):
                        os.remove(p)
        baseline_metric = None
        if self.cfg.baseline:
            bl = self.build_baseline(b["i0"], b["N"], out_dir)
            if self.cfg.verify:
                baseline_metric = self.evaluate(bl["sfz_path"], bl["key"], bl["N"] / self.sr, bl["loop_start"] / self.sr)
                self.log.append(f"baseline crossfade loop score={baseline_metric['score']:.3f}")
            b["baseline"] = bl
        result = LoopResult(self.name, b["sfz_path"], b["wav_paths"], self.seg.klass, b["key"], float(b["cents"]),
                            self.f0_nom, float(self.model.B), b["loop_start"], b["loop_end"], b["N"],
                            b["n0"] / self.sr, b["stages"], b["residual"], self.model.K, b["duration_s"],
                            len(self.x) / self.sr, b["size"], total_audio_s=b["total_s"], metric=b["metric"],
                            baseline_metric=baseline_metric,
                            candidates=[dict(i0=t["i0"], N=t["N"], cost=t["search_cost"],
                                             score=(t["metric"] or {}).get("score")) for t in tried],
                            info=dict(log=self.log, mode=b.get("blend"), nudge_cents_max=b["nudge_cents_max"], fit_err=b["fit_err"],
                                      fileg=b["fileg"], taus=b["taus"], release=b["release"],
                                      noise_rms=b["noise_rms"], baseline=b.get("baseline", {}).get("sfz_path")))
        with open(os.path.join(out_dir, f"{self.name}.json"), "w") as f:
            json.dump(asdict(result), f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
        return result


def process_sample(path: str, out_dir: str, cfg: LoopConfig | None = None) -> LoopResult:
    cfg = cfg or LoopConfig()
    if cfg.method == "laroche":
        from .laroche import LarocheLooper

        return LarocheLooper(path, cfg).run(out_dir)
    if cfg.method == "auto" and cfg.verify:
        # run both methods and keep the one Metric B prefers
        import shutil
        from dataclasses import replace

        from .laroche import LarocheLooper

        tmp = os.path.join(out_dir, "_auto_laroche")
        os.makedirs(tmp, exist_ok=True)
        r_l = LarocheLooper(path, replace(cfg, method="laroche")).run(tmp)
        r_h = SampleLooper(path, replace(cfg, method="hybrid")).run(out_dir)
        s_l = (r_l.metric or {}).get("score", -1.0)
        s_h = (r_h.metric or {}).get("score", -1.0)
        if r_l.klass != "oneshot" and s_l > s_h:
            # promote the laroche files over the hybrid ones
            for p in r_h.wav_paths + [r_h.sfz_path]:
                if os.path.exists(p):
                    os.remove(p)
            moved = []
            for p in r_l.wav_paths + [r_l.sfz_path, os.path.splitext(r_l.sfz_path)[0] + ".json"]:
                dst = os.path.join(out_dir, os.path.basename(p))
                shutil.move(p, dst)
                moved.append(dst)
            r_l.sfz_path = moved[-2]
            r_l.wav_paths = moved[:-2]
            if r_l.info.get("baseline"):
                bl = r_l.info["baseline"]
                for p in [bl, bl[:-4] + "." + cfg.out_format]:
                    if os.path.exists(p):
                        shutil.move(p, os.path.join(out_dir, os.path.basename(p)))
                r_l.info["baseline"] = os.path.join(out_dir, os.path.basename(bl))
            r_l.info["log"] = [f"auto: laroche {s_l:.3f} > hybrid {s_h:.3f}"] + r_l.info.get("log", [])
            shutil.rmtree(tmp, ignore_errors=True)
            return r_l
        r_h.info["log"] = [f"auto: hybrid {s_h:.3f} >= laroche {s_l:.3f}"] + r_h.info.get("log", [])
        shutil.rmtree(tmp, ignore_errors=True)
        return r_h
    return SampleLooper(path, cfg).run(out_dir)
