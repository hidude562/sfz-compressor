"""dctloop backend: the loop is built by ``dctloop`` (reconstruction on a loop-periodic grid, DCT or
DFT basis, harmonic locking) and wrapped into a playable SFZ recreation of the whole note:

    original attack  ->  short cross-fade  ->  dctloop loop (exactly periodic by construction)

plus the envelope machinery of the other methods (constant / exponential regions of the *same*
file, release from the recorded tail), the crossfade-loop baseline, the sfizz render and Metric B.
The loop is circularly rotated to the best phase match with the original at the join and
level-matched there, so the junction is coherent for both bases (the DFT loop is already aligned
with the central analysis frame; the DCT loop is a palindrome and needs the rotation).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import numpy as np

from . import dsp
from .envelope import fit_envelope_components, fit_fileg, fit_release_time, tau_to_sfz_time
from .harmonic import estimate_f0_spectral
from .looper import LoopResult, SampleLooper, _classic_crossfade_loop


def _import_dctloop():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    import dctloop

    return dctloop


class DctLoopLooper(SampleLooper):
    """SampleLooper whose loop comes from dctloop (no partial tracking needed)."""

    def analyse_light(self) -> None:
        x, sr = self.x, self.sr
        hint_hz = dsp.midi_to_hz(dsp.name_to_midi(self.cfg.note_hint)) if self.cfg.note_hint else None
        seg0 = dsp.segment_note(x, sr, f0_hint=hint_hz)
        f0_nom, sal = estimate_f0_spectral(x, sr, seg0.attack_end, max(seg0.release_onset, seg0.attack_end + sr // 10),
                                           hint_hz=hint_hz)
        self.pitched = sal > 2.0 and f0_nom > 0
        self.f0_nom = f0_nom
        self.voiced_frac = sal / 6.0
        budget = self.cfg.max_total_s
        over_budget = budget is not None and (seg0.end - seg0.onset) / sr > budget
        self.seg = dsp.segment_note(x, sr, f0_hint=f0_nom if self.pitched else None,
                                    force_loop=bool(over_budget and self.pitched))
        self.model = None
        self.res = self.cfg.resolved(int(np.floor((sr / 2 - 200) / max(f0_nom, 20.0))))
        self.log.append(f"class={self.seg.klass} pitched={self.pitched} f0~{f0_nom:.2f}Hz salience={sal:.1f} "
                        f"onset={self.seg.onset/sr:.3f}s attack_end={self.seg.attack_end/sr:.3f}s "
                        f"release_onset={self.seg.release_onset/sr:.3f}s end={self.seg.end/sr:.3f}s")

    # ------------------------------------------------------------ helpers
    def _target_seconds(self, join: int) -> float:
        q = self.res["q"]
        target = self.cfg.loop_seconds if self.cfg.loop_seconds else (0.25 + 1.25 * q)
        if self.cfg.max_total_s is not None:
            # dctloop rounds the loop to whole periods (up to half a period longer): keep one period of margin
            period = 1.0 / max(self.f0_nom, 20.0)
            # fit_loop_length may pick up to two periods more than the target for a better grid fit
            target = min(target, self.cfg.max_total_s - (join - self.seg.onset) / self.sr - 0.015 - 2.5 * period)
        body_s = (self.seg.release_onset - self.seg.attack_end) / self.sr
        return float(max(0.005, min(target, body_s / 2 * 0.99)))

    @staticmethod
    def _phase_align(loop: np.ndarray, ref: np.ndarray, sr: int) -> np.ndarray:
        """Keep the loop's grid magnitudes, take the phases of the original *at the join*, per channel.

        ``ref`` holds 2h samples of the original centred on the join (h <= L).  A zero-phase Hann
        frame centred there, zero-padded to a fine FFT, gives every loop-grid frequency the phase of
        the (possibly off-grid) partial at the join itself, so the loop's sample 0 lines up with the
        original even for inharmonic partials."""
        from scipy.signal.windows import get_window

        L, C = loop.shape
        n = len(ref)
        h = n // 2
        w = get_window("hann", 2 * h, fftbins=True)[:, None]
        fr = ref[: 2 * h] * w
        nfft = 1 << int(np.ceil(np.log2(max(8 * 2 * L, 2 * h))))
        buf = np.zeros((nfft, C))
        buf[:h] = fr[h:]                       # centre (the join) -> index 0
        buf[nfft - h:] = fr[:h]
        X = np.fft.rfft(buf, axis=0)
        fm = np.arange(L // 2 + 1) * sr / L    # loop grid frequencies
        idx = np.clip(np.round(fm / (sr / nfft)).astype(int), 0, X.shape[0] - 1)
        ph = np.angle(X[idx])
        Y = np.fft.rfft(loop, axis=0)
        Z = np.abs(Y) * np.exp(1j * ph)
        Z[0] = 0.0
        if L % 2 == 0:
            Z[-1] = Z[-1].real
        return np.fft.irfft(Z, n=L, axis=0)

    @staticmethod
    def _smooth_log_spectrum(P: np.ndarray, fr: np.ndarray, octaves: float = 1 / 3, min_bins: int = 17) -> np.ndarray:
        """Power spectrum smoothed to a broad spectral envelope: at least ``min_bins`` wide in Hz (so
        the harmonic fine structure is never resolved) and a constant fraction of an octave above that."""
        P = dsp.smooth(P, min_bins | 1)
        lf = np.log2(np.maximum(fr, 20.0))
        grid = np.linspace(lf[1], lf[-1], 600)
        lp = np.interp(grid, lf, 10 * np.log10(P + 1e-20))
        n = max(3, int(round(octaves / (grid[1] - grid[0]))) | 1)
        lp = dsp.smooth(lp, n)
        return 10 ** (np.interp(lf, grid, lp) / 10)

    def _morph_tail(self, tail: np.ndarray, loop: np.ndarray, max_db: float = 9.0) -> np.ndarray:
        """EQ-morph the recording's tail (M samples before the join) towards the loop's spectrum.

        A per-band gain (loop LTAS over the tail's last 40 ms, 1/6-octave smoothed, power-neutral,
        clipped to +-max_db) is applied through an STFT with a raised-cosine ramp from 0 (start of
        the tail: untouched) to 1 (the join), per channel.  The loop is not modified.
        """
        sr = self.sr
        M, C = tail.shape
        n_fft, hop = 2048, 512
        if M < 3 * n_fft // 2:
            return tail
        w = np.hanning(n_fft + 1)[:-1]
        fr = np.fft.rfftfreq(n_fft, 1 / sr)
        out = np.zeros_like(tail)
        norm = np.zeros(M)
        n_frames = 1 + (M - n_fft) // hop
        for c in range(C):
            # target: loop LTAS; source: the last 40 ms of the tail (a few frames), same resolution
            lt = np.tile(loop[:, c], max(1, int(np.ceil(2 * n_fft / len(loop))) + 1))
            starts = range(0, len(lt) - n_fft + 1, hop)
            T = np.mean([np.abs(np.fft.rfft(lt[s: s + n_fft] * w)) ** 2 for s in starts], axis=0)
            src_frames = [np.abs(np.fft.rfft(tail[s: s + n_fft, c] * w)) ** 2
                          for s in range(max(0, M - n_fft - 2 * hop), M - n_fft + 1, hop)] or \
                         [np.abs(np.fft.rfft(tail[M - n_fft:, c] * w)) ** 2]
            S = np.mean(src_frames, axis=0)
            Ts, Ss = self._smooth_log_spectrum(T, fr), self._smooth_log_spectrum(S, fr)
            G = np.sqrt(Ts / (Ss + 1e-20))
            G = np.clip(G, 10 ** (-max_db / 20), 10 ** (max_db / 20))
            G *= np.sqrt(np.sum(S) / (np.sum(S * G ** 2) + 1e-20))   # power-neutral on the source
            logG = np.log(G)
            for i in range(n_frames):
                s0 = i * hop
                ramp = 0.5 - 0.5 * np.cos(np.pi * min(1.0, (s0 + n_fft / 2) / M))   # 0 -> 1 towards the join
                X = np.fft.rfft(tail[s0: s0 + n_fft, c] * w)
                y = np.fft.irfft(X * np.exp(ramp * logG), n=n_fft) * w
                out[s0: s0 + n_fft, c] += y
                if c == 0:
                    norm[s0: s0 + n_fft] += w ** 2
        good = norm > 1e-3
        out[good] /= norm[good, None]
        # the first/last partial frames of the OLA are not fully covered: keep the recording there
        out[~good] = tail[~good]
        return out

    def _junction_score(self, loop: np.ndarray, join: int) -> tuple[float, float, float]:
        """(score, dip_db, transient_db) of the attack -> loop junction for a candidate loop alignment.

        dip: level of the cross-fade mix relative to the original in the same span (cancellation < 0);
        transient: high-passed spectral-flux novelty at the join relative to the 90th percentile of the
        following 300 ms, minus the same for the original."""
        from .metric import _novelty

        seg, sr, x = self.seg, self.sr, self.x
        L = len(loop)
        Xc = int(min(int((0.06 if seg.klass == "decay" else 0.01) * sr), join - seg.onset, L // 2))
        Xc = max(Xc, 1)
        w_j = int(np.clip(4 * sr / max(self.f0_nom, 20.0), 0.02 * sr, 0.05 * sr))
        g = np.sqrt(np.mean(x[join: join + w_j] ** 2)) / (np.sqrt(np.mean(loop ** 2)) + 1e-12)
        lp = loop * float(np.clip(g, 0.1, 10.0))
        w = dsp.raised_cosine(Xc)[:, None]
        mix = x[join - Xc: join] * (1 - w) + lp[L - Xc:] * w
        dip = 20 * np.log10((np.sqrt(np.mean(mix ** 2)) + 1e-12) / (np.sqrt(np.mean(x[join - Xc: join] ** 2)) + 1e-12))
        a0 = max(seg.onset, join - int(0.3 * sr))
        n_after = min(L, int(0.4 * sr))
        file_seg = np.concatenate([x[a0: join - Xc], mix, np.tile(lp, (2, 1))[:n_after]], axis=0)
        orig_seg = x[a0: a0 + len(file_seg)]
        hop = 256
        k = max(3, int(round(0.05 * sr / hop)) | 1)

        def spike(sig):
            nov = _novelty(dsp.to_mono(sig), sr, hop)
            hp = np.maximum(nov - dsp.smooth(nov, k), 0)
            j = (join - a0) // hop
            ref = np.percentile(hp[j + 3: j + 3 + int(0.3 * sr / hop)], 90) + 1e-9
            return 20 * np.log10(hp[max(0, j - 2): j + 3].max() / ref + 1e-9)

        tr = float(np.clip(spike(file_seg) - spike(orig_seg), -20, 40))
        score = max(0.0, -dip) + max(0.0, tr) / 4.0
        return float(score), float(dip), tr

    @staticmethod
    def _best_rotation(loop: np.ndarray, target: np.ndarray) -> int:
        """Circular shift tau maximising the correlation of loop[(tau+n) mod L] with target[n]."""
        from scipy.signal import correlate

        lm = loop.mean(axis=1) if loop.ndim == 2 else loop
        tm = target.mean(axis=1) if target.ndim == 2 else target
        L = len(lm)
        W = min(len(tm), L)
        tm = tm[:W]
        tl = np.concatenate([lm, lm[:W]])
        c = correlate(tl, tm, mode="valid", method="fft")  # c[tau] = sum_n tl[tau+n] tm[n]
        return int(np.argmax(c[:L]))

    def _assemble(self, loop: np.ndarray, info: dict, basis: str, join: int, out_dir: str, tag: str = "") -> dict:
        seg, sr, x = self.seg, self.sr, self.x
        C = x.shape[1]
        L = len(loop)
        # line the loop up with the original at the join.  A DFT loop gets the original's phase at
        # every grid bin measured at the join (a 2L Hann frame starting there: its even bins carry the
        # phase at the frame start), so every partial - harmonic or not - is coherent through the
        # cross-fade.  A DCT loop (palindrome) can only be rotated to the best overall match.
        cands = []
        tau_r = self._best_rotation(loop, x[join: join + min(L, int(0.1 * sr))])
        cands.append(("rotate", np.roll(loop, -tau_r, axis=0), tau_r))
        h = int(min(L, join - seg.onset, len(x) - join))
        if basis == "dft" and h >= int(4 * sr / max(self.f0_nom, 20.0)):
            cands.append(("phase", self._phase_align(loop, x[join - h: join + h], sr), 0))
        # keep the alignment with the least cancellation in the cross-fade region and the smallest
        # transient at the join (measured on the assembled file, no render needed)
        best_c = None
        for name, lp_c, tau_c in cands:
            sc = self._junction_score(lp_c, join)
            if best_c is None or sc[0] < best_c[0]:
                best_c = (sc[0], name, lp_c, tau_c, sc)
        _, align, loop, tau, jsc = best_c
        self.log.append(f"join alignment: " + ", ".join(f"{n}={self._junction_score(l, join)[0]:.2f}" for n, l, _ in cands)
                        + f" -> {align} (dip {jsc[1]:+.1f} dB, transient {jsc[2]:+.1f} dB)")
        # level-match the (stationary) loop to the recording *at the join*: a short window there,
        # not the average of the next loop length (which is far below the join level on a decaying note)
        f0_here = float(info.get("f0_used") or self.f0_nom or 100.0)
        w_join = int(np.clip(4 * sr / max(f0_here, 20.0), 0.02 * sr, 0.05 * sr))
        ref = x[join: join + w_join]
        # per-channel (not mono-sum) RMS: the loop's inter-channel phases differ from the original's
        g = np.sqrt(np.mean(ref ** 2)) / (np.sqrt(np.mean(loop ** 2)) + 1e-12)
        g = float(np.clip(g, 0.1, 10.0))
        loop = loop * g
        # phase-aligned cross-fade into the loop; longer for decaying notes whose timbre is still moving
        X = int(min(int((0.06 if seg.klass == "decay" else 0.01) * sr), join - seg.onset, L // 2))
        X = max(X, 1)
        w = dsp.raised_cosine(X)[:, None]
        pre = loop[L - X:] if X > 0 else loop[:0]
        # EQ-morph the recording's tail towards the loop's spectrum (the loop itself is untouched)
        Mt = int(min(self.cfg.tail_morph_s * sr, 0.7 * (join - seg.onset)))
        rec = x[seg.onset: join].copy()
        morphed = False
        if self.cfg.tail_morph and Mt >= int(0.08 * sr):
            rec[-Mt:] = self._morph_tail(rec[-Mt:], loop)
            morphed = True
        head = rec[: len(rec) - X]
        xf = rec[len(rec) - X:] * (1 - w) + pre * w
        out = np.concatenate([head, xf, loop], axis=0)
        nf = min(len(out), int(0.002 * sr))
        if nf > 1:
            out[:nf] *= dsp.raised_cosine(nf)[:, None]
        loop_start = join - seg.onset
        loop_end = loop_start + L - 1
        t_hold = loop_start / sr
        # broadband level from the join onwards -> constant / exponential regions of the same file
        et, edb = dsp.rms_envelope(x, sr, win=0.02, hop=0.005)
        sel = (et >= join) & (et <= max(seg.release_onset, join + int(0.05 * sr)))
        lvl = 10 ** (edb[sel] / 20) if sel.sum() >= 3 else np.ones(3)
        tt = (et[sel] - join) / sr if sel.sum() >= 3 else np.array([0.0, 0.05, 0.1])
        n_norm = max(1, int(round(w_join / (0.005 * sr))))   # the same window as the level match
        lvl = lvl / (lvl[: min(n_norm, len(lvl))].mean() + 1e-12)
        klass = seg.klass
        # regions of the same file are free in size: decaying notes get up to three exponentials so the
        # fast initial decay (piano "prompt sound") is followed as well as the long tail
        stages = 3 if klass == "decay" else max(1, min(self.res["stages"], 2))
        # non-negative: every component must be a region with a positive gain (a rising level after the
        # join cannot be expressed, so the loop simply holds the join level)
        comps, fit_err, base_err = fit_envelope_components(lvl[None, None, :], tt, klass, stages, nonneg=True)
        tot = sum(float(c.coef[0, 0]) for c in comps)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        gdb = 0.0
        if peak > 0.98:
            out *= 0.98 / peak
            gdb = 20 * np.log10(peak / 0.98)
        rel = fit_release_time(seg.env_t, seg.env_db, sr, seg.release_onset, seg.end,
                               default=0.4 if klass == "sustain" else 0.25)
        regions = []
        for c in comps:
            frac = float(c.coef[0, 0]) / tot if tot > 0 else 1.0
            if frac <= 1e-3:  # a decaying note's slow tail is small relative to the join level: keep it
                continue
            env = "ampeg_sustain=100" if c.kind == "const" else \
                f"ampeg_hold={t_hold:.5f} ampeg_decay={tau_to_sfz_time(c.tau):.4f} ampeg_sustain=0"
            regions.append(dict(kind=c.kind, env=env, gain_db=gdb + 20 * np.log10(frac)))
        if not regions:
            regions.append(dict(kind="const", env="ampeg_sustain=100", gain_db=gdb))
        # decaying notes darken: fit a low-pass envelope so the loop's centroid follows the original's
        fileg = None
        if klass == "decay" and self.cfg.fileg:
            fileg = self._fit_darkening(loop, join)
        f0 = info.get("f0_used") or self.f0_nom
        key = int(round(dsp.hz_to_midi(f0)))
        cents = 100 * (dsp.hz_to_midi(f0) - key)
        p = os.path.join(out_dir, f"{self.name}{tag}.{self.cfg.out_format}")
        dsp.write_audio(p, out, sr, subtype="PCM_16")
        sfz_path = os.path.join(out_dir, f"{self.name}{tag}.sfz")
        met = info.get("dct_metrics", {})
        with open(sfz_path, "w") as f:
            f.write(f"// dctloop {basis} loop | q={self.res['q']:.2f} class={klass} f0={f0:.2f}Hz "
                    f"({cents:+.1f} cents from key {key}) pitch={info.get('pitch', {}).get('method', '?')}\n")
            f.write(f"// loop {loop_start}..{loop_end} ({L} samples = {L/sr*1000:.1f} ms, K={info.get('K')} periods, "
                    f"grid {info.get('grid_cents', 0):+.2f} c, frames={info.get('frames')}, lock={info.get('lock_width')}, "
                    f"rotation={tau} samples, join gain={20*np.log10(g):+.1f} dB, "
                    f"tail morph={'%.0f ms' % (Mt / sr * 1000) if morphed else 'off'})\n")
            if met:
                f.write(f"// dctloop metrics: seam x{met.get('seam_flux_ratio', 0):.2f} p95 x{met.get('p95_flux_ratio', 0):.2f} "
                        f"ltas {met.get('ltas_mean_abs_db', 0):.2f}/{met.get('ltas_max_abs_db', 0):.2f} dB "
                        f"wah {met.get('max_harmonic_am_db', 0):.1f} dB\n")
            if fileg is not None:
                f.write(f"// darkening: cutoff {fileg.cutoff:.0f} Hz + {fileg.depth_cents:.0f} cents decaying with T={fileg.decay_time:.2f} s\n")
            f.write("<global> amp_veltrack=0\n")
            for r in regions:
                f.write(f"<region> sample={os.path.basename(p)} pitch_keycenter={key} lokey={key} hikey={key}  // {r['kind']}\n"
                        f"  loop_mode=loop_continuous loop_start={loop_start} loop_end={loop_end}\n"
                        f"  volume={r['gain_db']:.3f} {r['env']} ampeg_release={rel:.4f}\n")
                if fileg is not None:
                    f.write(f"  {fileg.opcodes(t_hold)}\n")
        return dict(sfz_path=sfz_path, wav_paths=[p], key=key, cents=cents, loop_start=loop_start, loop_end=loop_end,
                    N=L, n0=join, i0=0, iN=0, stages=len(regions), residual=False, total_s=len(out) / sr,
                    nudge_cents_max=abs(float(info.get("grid_cents", 0.0))), nudge_cents_w=abs(float(info.get("grid_cents", 0.0))),
                    fit_err=fit_err, base_err=base_err, fileg=(asdict(fileg) if fileg else None), noise_rms=0.0,
                    size=os.path.getsize(p),
                    duration_s=len(out) / sr, release=rel, taus=[c.tau for c in comps], blend=f"dctloop-{basis}",
                    rotation=tau, join_gain_db=float(20 * np.log10(g)), dct_info={k: v for k, v in info.items()
                                                                                  if k not in ("pitch",)})

    def _fit_darkening(self, loop: np.ndarray, join: int):
        """fileg low-pass envelope so the loop's spectral centroid follows the original's from the join on."""
        seg, sr, x = self.seg, self.sr, self.x
        lm = dsp.to_mono(loop)
        A = np.abs(np.fft.rfft(lm * np.hanning(len(lm))))
        fr = np.fft.rfftfreq(len(lm), 1 / sr)
        keep = (fr > 20) & (fr < sr / 2 - 100)
        n_fft = 2048
        hop = int(0.02 * sr)
        m = dsp.to_mono(x)
        starts = np.arange(join, max(join + 1, seg.release_onset - n_fft), hop)
        if len(starts) < 6:
            return None
        w = np.hanning(n_fft)
        fb = np.fft.rfftfreq(n_fft, 1 / sr)
        cen = np.empty(len(starts))
        for i, s0 in enumerate(starts):
            P = np.abs(np.fft.rfft(m[s0: s0 + n_fft] * w)) ** 2
            cen[i] = (P * fb).sum() / (P.sum() + 1e-20)
        cen = dsp.smooth(cen, 5)
        t = (starts - join) / sr
        return fit_fileg(A[keep], fr[keep], cen, t)

    def _baseline(self, join: int, L: int, out_dir: str) -> dict:
        seg, sr, x = self.seg, self.sr, self.x
        y, N = _classic_crossfade_loop(x, sr, seg, join, L, self.f0_nom)
        p = os.path.join(out_dir, f"{self.name}_xfade.{self.cfg.out_format}")
        dsp.write_audio(p, y, sr)
        key = int(round(dsp.hz_to_midi(self.f0_nom)))
        loop_start = join - seg.onset
        rel = fit_release_time(seg.env_t, seg.env_db, sr, seg.release_onset, seg.end)
        sfz_path = os.path.join(out_dir, f"{self.name}_xfade.sfz")
        with open(sfz_path, "w") as f:
            f.write("// baseline: classic equal-power crossfade loop\n<global> amp_veltrack=0\n")
            f.write(f"<region> sample={os.path.basename(p)} pitch_keycenter={key} lokey={key} hikey={key} "
                    f"loop_mode=loop_continuous loop_start={loop_start} loop_end={loop_start + N - 1} "
                    f"ampeg_sustain=100 ampeg_release={rel:.4f}\n")
        return dict(sfz_path=sfz_path, wav_paths=[p], key=key, loop_start=loop_start, N=N, size=os.path.getsize(p))

    # ------------------------------------------------------------ driver
    def run(self, out_dir: str) -> LoopResult:
        os.makedirs(out_dir, exist_ok=True)
        self.analyse_light()
        if not self.pitched or self.seg.klass == "oneshot":
            return self._write_oneshot(out_dir)
        dl = _import_dctloop()
        seg, sr, x = self.seg, self.sr, self.x
        q = self.res["q"]
        # the loop joins right after the onset transient (a little later at high q without a budget).
        # Under a budget a decaying note may do better keeping more of its recorded early decay and
        # looping less: several join points are tried and Metric B arbitrates.
        join0 = seg.attack_end + (0 if self.cfg.max_total_s is not None else int(0.3 * q * sr))
        join0 = int(min(join0, max(seg.attack_end, seg.release_onset - int(0.05 * sr))))
        joins = [join0]
        if self.cfg.max_total_s is not None and self.cfg.verify:
            B = self.cfg.max_total_s * sr
            for frac in (0.25, 0.4):
                jn = int(seg.onset + frac * B)
                if jn > join0 + int(0.02 * sr) and jn < seg.release_onset - int(0.1 * sr):
                    joins.append(jn)
        hint = dl.note_from_name(self.name) or self.f0_nom
        bases = ["dct", "dft"] if self.cfg.dct_basis == "auto" else [self.cfg.dct_basis]
        if not self.cfg.verify:
            bases = bases[:1]
        tried = []
        best = None
        ci = 0
        for join in joins:
            target = self._target_seconds(join)
            if seg.klass == "decay":
                # a decaying note is not stationary: analyse the shortest usable stretch right after the
                # join (2-3 loop lengths), level-detrended, so the loop carries the join's timbre and the
                # envelope / darkening opcodes carry the evolution
                n_seg = int(min(seg.release_onset - join, max(2.02 * target * sr, min(3 * target * sr, 1.5 * sr))))
                n_seg = max(n_seg, min(seg.release_onset - join, int(0.2 * sr)))
                body = x[join: join + n_seg].copy()
                et, edb = dsp.rms_envelope(body, sr, win=0.02, hop=0.005)
                slope_db = 0.0
                if len(et) >= 4:
                    A_ = np.vstack([et / sr, np.ones_like(et, dtype=float)]).T
                    slope_db = float(np.linalg.lstsq(A_, edb, rcond=None)[0][0])
                    body *= (10 ** (-slope_db * (np.arange(len(body)) / sr) / 20))[:, None]
                self.log.append(f"decay: join {join/sr*1000:.0f} ms, dctloop analyses {n_seg/sr:.2f}s after it, "
                                f"detrended by {slope_db:+.1f} dB/s")
            else:
                body = x[seg.attack_end: seg.release_onset]
            for basis in bases:
                loop, info = dl.loop_signal(body, sr, target, basis=basis, hint=hint, use_hint=True,
                                            lock=self.cfg.dct_lock, seed=self.cfg.seed)
                seg_used = body[info["analysis_offset"]: info["analysis_offset"] + info["N"]]
                try:
                    info["dct_metrics"] = dl.measure(seg_used, sr, loop, [v for v in info["f0"] if np.isfinite(v)] or None)
                except Exception:  # metrics are informative only
                    info["dct_metrics"] = {}
                b = self._assemble(loop, info, basis, join, out_dir, tag="" if ci == 0 else f"_c{ci}")
                ci += 1
                b["search_cost"] = 0.0
                b["basis"] = basis
                if self.cfg.verify:
                    b["metric"] = self.evaluate(b["sfz_path"], b["key"], b["N"] / sr, b["loop_start"] / sr)
                    score = b["metric"]["score"]
                else:
                    b["metric"] = None
                    score = 0.0
                tried.append(b)
                m = info["dct_metrics"]
                self.log.append(f"dctloop {basis} join={join/sr*1000:.0f}ms: f0={info['f0_used']:.2f}Hz/{info.get('pitch', {}).get('method', '?')} "
                                f"L={info['L']} ({info['seconds']*1000:.1f} ms, K={info['K']}, grid {info['grid_cents']:+.2f}c) "
                                f"frames={info.get('frames')} seam x{m.get('seam_flux_ratio', 0):.2f} wah {m.get('max_harmonic_am_db', 0):.1f}dB "
                                f"rotation={b['rotation']} join {b['join_gain_db']:+.1f}dB score={score:.3f} total={b['total_s']:.3f}s"
                                + (" (shortened)" if info.get("shortened") else ""))
                if best is None or score > best[0]:
                    best = (score, b)
        _, b = best
        canon_sfz = os.path.join(out_dir, f"{self.name}.sfz")
        if b["sfz_path"] != canon_sfz:
            # promote the winner to the canonical names
            for t in tried:
                if t is not b:
                    for p in t["wav_paths"] + [t["sfz_path"]]:
                        if os.path.exists(p):
                            os.remove(p)
            for src_key in ("sfz_path",):
                dst = canon_sfz
                txt = open(b["sfz_path"]).read()
                os.remove(b["sfz_path"])
            new_wav = os.path.join(out_dir, f"{self.name}.{self.cfg.out_format}")
            os.replace(b["wav_paths"][0], new_wav)
            txt = txt.replace(os.path.basename(b["wav_paths"][0]), os.path.basename(new_wav))
            with open(canon_sfz, "w") as f:
                f.write(txt)
            b["sfz_path"], b["wav_paths"] = canon_sfz, [new_wav]
        else:
            for t in tried:
                if t is not b:
                    for p in t["wav_paths"] + [t["sfz_path"]]:
                        if os.path.exists(p):
                            os.remove(p)
        baseline_metric = None
        if self.cfg.baseline:
            bl = self._baseline(join, b["N"], out_dir)
            if self.cfg.verify:
                baseline_metric = self.evaluate(bl["sfz_path"], bl["key"], bl["N"] / sr, bl["loop_start"] / sr)
                self.log.append(f"baseline crossfade loop score={baseline_metric['score']:.3f}")
            b["baseline"] = bl
        result = LoopResult(self.name, b["sfz_path"], b["wav_paths"], seg.klass, b["key"], float(b["cents"]),
                            float(b["dct_info"].get("f0_used") or self.f0_nom), 0.0, b["loop_start"], b["loop_end"],
                            b["N"], b["n0"] / sr, b["stages"], False, 0, b["duration_s"], len(x) / sr, b["size"],
                            total_audio_s=b["total_s"], metric=b["metric"], baseline_metric=baseline_metric,
                            candidates=[dict(basis=t["basis"], N=t["N"], score=(t["metric"] or {}).get("score")) for t in tried],
                            info=dict(log=self.log, mode=b["blend"], nudge_cents_max=b["nudge_cents_max"],
                                      fit_err=b["fit_err"], fileg=None, taus=b["taus"], release=b["release"],
                                      noise_rms=0.0, baseline=b.get("baseline", {}).get("sfz_path"),
                                      dctloop=b["dct_info"], rotation=b["rotation"], join_gain_db=b["join_gain_db"]))
        with open(os.path.join(out_dir, f"{self.name}.json"), "w") as f:
            json.dump(asdict(result), f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        return result
