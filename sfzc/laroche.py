"""Laroche-style loop-locked resynthesis (fork of the parametric path).

Implements the recommended algorithm of the "Reconstructing sustain segments into exactly periodic
loops" report:

* partials frozen to their loop-region statistics (median frequency, detrended mean amplitude,
  analysed phase at the loop start) and **loop-locked**: every partial completes an integer number
  of cycles in L (nearest grid frequency k·fs/L == Laroche's minimal nudge for a stationary partial);
* L chosen by an energy- and JND-weighted detuning cost (+ vibrato-cycle term, + budget), aiming
  for ~166 fundamental periods when the budget allows;
* oscillator-bank resynthesis -> exactly periodic by construction, coherent envelope stages;
* the residual re-synthesised as filtered noise in a **separate loop with a different length**,
  so harmonic and noise repetition periods never coincide (combined period = lcm);
* DC removal, RMS match, seam diagnostics, optional LFO / loop_crossfade / round-robin opcodes;
* optional Stage-3 refinement of partial and noise-band gains against a multi-resolution STFT loss.

Notes that has vibrato/tremolo (parameter tracks that move) are handed to the tracked path of
``SampleLooper`` (parameter-domain closing keeps the movement), unless ``frozen='on'``.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from . import dsp
from .envelope import EnvComponent, fit_envelope_components, fit_release_time, make_exact_at_start, tau_to_sfz_time  # noqa
from .harmonic import synth_partials
from .looper import LoopResult, SampleLooper, _detrend_log_span, _periodic_noise, _wrap


def _jnd_cents(f: np.ndarray) -> np.ndarray:
    """Frequency JND in cents: ~3 Hz below 500 Hz, ~0.6 % (10 cents) above."""
    f = np.maximum(np.asarray(f, float), 20.0)
    return 1200 * np.log2(1 + np.maximum(3.0, 0.006 * f) / f)


class LarocheLooper(SampleLooper):
    # ----------------------------------------------------------- modulation
    def _modulation(self) -> dict:
        """Vibrato / tremolo statistics of the body from the model's f0 and amplitude tracks."""
        m, seg, sr = self.model, self.seg, self.sr
        i_a, i_r = m.frame_at(seg.attack_end), m.frame_at(seg.release_onset)
        f0 = m.f0.mean(axis=0)[i_a: max(i_a + 2, i_r)]
        lvl = 20 * np.log10(np.sqrt((m.amp_mean ** 2).sum(axis=0))[i_a: max(i_a + 2, i_r)] + 1e-9)
        fr = sr / m.hop
        out = dict(vib_rate=0.0, vib_cents=0.0, trem_db=0.0)
        if len(f0) < 16:
            return out
        cents = 1200 * np.log2(np.maximum(f0, 1e-3) / np.median(f0))
        k = max(3, int(round(0.4 * fr)) | 1)
        c_hp = cents - dsp.smooth(cents, k)
        l_hp = lvl - dsp.smooth(lvl, k)
        n = 1 << int(np.ceil(np.log2(len(c_hp) * 4)))
        S = np.abs(np.fft.rfft(c_hp * np.hanning(len(c_hp)), n=n)) ** 2
        fb = np.fft.rfftfreq(n, 1 / fr)
        band = (fb >= 3) & (fb <= 9)
        if band.any() and S[band].max() > 0:
            out["vib_rate"] = float(fb[band][np.argmax(S[band])])
        out["vib_cents"] = float(np.std(c_hp))
        out["trem_db"] = float(np.std(l_hp))
        return out

    # ----------------------------------------------------------- L selection
    def _laroche_candidates(self, n_files: int, mod: dict) -> list[tuple[float, int, int]]:
        model, seg, sr, cfg = self.model, self.seg, self.sr, self.cfg
        q = self.res["q"]
        i_a, i_r = model.frame_at(seg.attack_end), model.frame_at(seg.release_onset)
        period = sr / self.f0_nom
        budget = None
        if cfg.max_total_s is not None:
            hard_attack = min(seg.attack_end, seg.onset + int(0.1 * sr) + int(2 * period))
            i_a = model.frame_at(max(hard_attack, seg.onset + int(0.02 * sr)))
            budget = int(cfg.max_total_s * sr)
        N_min = int(max(2 * period, 0.005 * sr))
        # loop start: try a few positions after the attack and keep the one whose body envelope is
        # best explained by the coherent envelope stages (a late start skips the non-exponential
        # prompt decay of struck/plucked notes); the budget limits how much attack audio we keep
        i0_best, err_best = i_a, np.inf
        for dt in (0.0, 0.1, 0.2, 0.35, 0.5):
            i0c = model.frame_at(int(model.centers[i_a]) + int(dt * q * sr))
            if i0c > i_r - 4:
                break
            if budget is not None and (int(model.centers[i0c]) - seg.onset) + n_files * N_min > budget:
                break
            body_c = slice(i0c, max(i0c + 2, i_r + 1))
            _, err_c, _ = fit_envelope_components(model.amp[:, :, body_c], (model.centers[body_c] - model.centers[i0c]) / sr,
                                                  seg.klass, max(1, min(2, self.res["stages"])))
            if err_c < err_best - 0.05:
                i0_best, err_best = i0c, err_c
        i_a = i0_best
        n0 = int(model.centers[i_a])
        L_full = max(0, (i_r - i_a) * model.hop)
        N_max = int(N_min + (0.8 * L_full - N_min) * q ** 1.5)
        N_max = max(N_min, N_max // n_files)
        if budget is not None:
            N_max = max(N_min, min(N_max, (budget - (n0 - seg.onset)) // n_files))
        body = slice(i_a, max(i_a + 1, i_r + 1))
        # partial statistics over the body (per channel): median frequency, mean power
        f_i = np.median(model.freq[:, :, body], axis=2)            # (C, K)
        p_i = (model.amp[:, :, body] ** 2).mean(axis=2)             # (C, K)
        w = p_i / (p_i.sum() + 1e-20)
        jnd = _jnd_cents(f_i)
        target_N = cfg.target_periods * period
        m_min = max(2, int(np.ceil(N_min / period)))
        m_max = max(m_min, int(N_max // period))
        m_vals = np.unique(np.round(np.geomspace(m_min, m_max, 80)).astype(int))
        cands = []
        for m in m_vals:
            N = int(round(m * period))
            k = np.maximum(np.round(f_i * N / sr), 1.0)
            cents = 1200 * np.log2((k * sr / N) / np.maximum(f_i, 1e-3))
            cost = float(np.sum(w * (cents / jnd) ** 2))
            # prefer lengths near the target period count (Laroche's ~166 periods); mild
            cost += 0.02 * abs(np.log(max(N, 1) / target_N))
            if mod["vib_rate"] > 0 and mod["vib_cents"] > 3.0:
                cyc = N * mod["vib_rate"] / sr
                cost += 0.5 * (mod["vib_cents"] / 10) * (cyc - round(cyc)) ** 2
            cands.append((cost, i_a, N))
        cands.sort(key=lambda r: r[0])
        # diversity: best per factor-2 length band, longest first, then best overall
        bands: dict[int, tuple[float, int, int]] = {}
        for r in cands:
            b = int(np.floor(np.log2(max(r[2], 1) / max(N_min, 1))))
            bands.setdefault(b, r)
        picked = [bands[b] for b in sorted(bands, reverse=True)]
        for r in cands:
            if len(picked) >= 8:
                break
            if all(abs(r[2] - p[2]) > 0.1 * p[2] for p in picked):
                picked.append(r)
        self.log.append(f"laroche L search ({n_files} files): {len(m_vals)} lengths in [{N_min/sr:.3f}, {N_max/sr:.3f}] s, "
                        f"target {target_N/sr:.2f} s, best cost {cands[0][0]:.3f} at {cands[0][2]/sr*1000:.0f} ms")
        return picked

    # --------------------------------------------------------------- build
    def build_laroche(self, i0: int, N: int, stages: int, residual: bool, out_dir: str, tag: str = "") -> dict:
        model, seg, sr, x, cfg = self.model, self.seg, self.sr, self.x, self.cfg
        C = x.shape[1]
        n0 = int(model.centers[i0])
        N = int(N)
        X_pre = max(1, int(min(model.M // 2, n0 - seg.onset, max(int(0.002 * sr), 1))))
        i1 = model.frame_at(n0 + N)
        loop_fr = slice(i0, max(i0 + 2, i1 + 1))
        # frozen partial parameters: detrended mean amplitude, median frequency, phase at n0
        a_tr = _detrend_log_span(model.amp[:, :, loop_fr], 0, max(2, i1 - i0)).mean(axis=2)   # (C, K)
        f_i = np.median(model.freq[:, :, loop_fr], axis=2)                                        # (C, K)
        # loop-lock: integer cycles in N (== Laroche nudge for a stationary partial)
        k_i = np.maximum(np.round(f_i * N / sr), 1.0)
        f_lock = k_i * sr / N
        detune = 1200 * np.log2(f_lock / np.maximum(f_i, 1e-3))
        wn = a_tr ** 2
        detune_w = float((np.abs(detune) * wn).sum() / (wn.sum() + 1e-20))
        phase0 = model.phase[:, :, i0]
        # envelope stages over the body
        i_r = model.frame_at(seg.release_onset)
        body = slice(i0, max(i0 + 2, i_r + 1))
        t_body = (model.centers[body] - n0) / sr
        klass = seg.klass
        comps, fit_err, base_err = fit_envelope_components(model.amp[:, :, body], t_body, klass, stages)
        make_exact_at_start(comps, a_tr)
        # oscillator bank, exactly periodic: constant tracks over [-X_pre, N)
        nS = X_pre + N
        Fq = np.repeat(f_lock[:, :, None], nS, axis=2)
        gains = np.ones(model.K)
        # ---- residual: filtered noise with its own loop length
        noise = None
        N_noise = 0
        if residual:
            rc = model.resid_centers
            fr = np.where((rc >= n0 - model.M) & (rc <= n0 + N + model.M))[0]
            if fr.size == 0:
                fr = np.array([int(np.argmin(np.abs(rc - n0)))])
            budget = cfg.max_total_s
            used = (n0 - seg.onset + N) + (len(comps) - 1) * nS + X_pre
            if budget is not None:
                N_noise = int(budget * sr) - used
            else:
                N_noise = int(round(N * 1.618))
            N_noise = int(np.clip(N_noise, min(N, int(0.1 * sr)), max(N, int(2.0 * sr))))
            if N_noise == N:
                N_noise = max(N - 1, 2)
            mags = []
            for q_i in range(2 if C == 2 else 1):
                mags.append(np.sqrt((model.resid_mag[q_i][:, fr].astype(np.float64) ** 2).mean(axis=1)))
            noise = np.zeros((X_pre + N_noise, C))
            idx = np.arange(-X_pre, N_noise) % N_noise
            for q_i, mag in enumerate(mags):
                per = _periodic_noise(mag, model.resid_freqs, model.resid_win_sq_sum, sr, N_noise, self.rng)
                sig = per[idx]
                if C == 2:
                    noise[:, 0] += sig
                    noise[:, 1] += sig if q_i == 0 else -sig
                else:
                    noise[:, 0] += sig
        # ---- optional refinement of partial / noise-band gains (Stage 3)
        refine_info = None
        if cfg.refine:
            from .refine import refine_gains, split_noise_bands

            tot = sum(c.coef for c in comps)                                    # (C, K) amplitude at t0
            n = np.arange(N)
            # exact mono mix of every synthesised partial (per-channel frequency, phase and level)
            parts = np.mean(tot[:, :, None] * np.cos(2 * np.pi * f_lock[:, :, None] * n / sr + phase0[:, :, None]),
                            axis=0)                                              # (K, N)
            nb = split_noise_bands(dsp.to_mono(noise[X_pre:]), sr) if noise is not None else np.zeros((0, N))
            if nb.shape[1] != N:  # noise loop length differs: tile / crop to N for the loss
                nb = np.tile(nb, (1, int(np.ceil(N / nb.shape[1]))))[:, :N]
            # target: the source around the loop start (a decaying note must not be matched to its average)
            tgt = dsp.to_mono(x[n0: min(seg.release_onset, n0 + max(3 * N, int(0.5 * sr)))])
            refine_info = refine_gains(parts, nb, tgt, sr)
            gains = refine_info["part_gains"]
            for c in comps:
                c.coef = c.coef * gains[None, :]
            if noise is not None and refine_info["noise_gains"].size:
                # apply band gains to the noise via FFT masks
                Y = np.fft.rfft(noise, axis=0)
                fb = np.fft.rfftfreq(len(noise), 1 / sr)
                edges = np.geomspace(60.0, sr / 2, len(refine_info["noise_gains"]) + 1)
                edges[0] = 0.0
                G = np.ones(len(fb))
                for b, gb in enumerate(refine_info["noise_gains"]):
                    G[(fb >= edges[b]) & (fb < edges[b + 1])] = gb
                noise = np.fft.irfft(Y * G[:, None], n=len(noise), axis=0)
            refine_info = {k: (float(v) if isinstance(v, float) else None) for k, v in refine_info.items()
                           if k in ("loss_before", "loss_after")}
        ys = []
        for comp in comps:
            amp_s = np.repeat(comp.coef[:, :, None], nS, axis=2)
            y = synth_partials(amp_s, Fq, phase0, sr, slope=None, phase_ref=X_pre)
            y -= y[X_pre:].mean(axis=0, keepdims=True)  # remove DC
            ys.append(y)
        # ---- assemble files
        w_in = dsp.raised_cosine(X_pre)[:, None]
        wavs = []
        head = x[seg.onset: n0 - X_pre].copy()
        xf = x[n0 - X_pre: n0] * (1 - w_in) + ys[0][:X_pre] * w_in
        out0 = np.concatenate([head, xf, ys[0][X_pre:]], axis=0)
        nf = min(len(out0), int(0.002 * sr))
        if nf > 1:
            out0[:nf] *= dsp.raised_cosine(nf)[:, None]
        wavs.append(out0)
        for y in ys[1:]:
            wavs.append(y.copy())
        if noise is not None:
            wavs.append(noise)
        loop_start = n0 - seg.onset
        loop_end = loop_start + N - 1
        t_hold = loop_start / sr
        pre_len = n0 - X_pre - seg.onset
        delay_s = (pre_len - cfg.delay_offset_samples + 0.5) / sr
        rel = fit_release_time(seg.env_t, seg.env_db, sr, seg.release_onset, seg.end,
                               default=0.4 if klass == "sustain" else 0.25)
        lfo = ("pitchlfo_freq=0.31 pitchlfo_depth=2 amplfo_freq=0.23 amplfo_depth=0.3" if cfg.lfo else "")
        xfade = f"loop_crossfade={cfg.loop_crossfade_s:.4f}" if cfg.loop_crossfade_s > 0 else ""

        def env_opcodes(comp, extra):
            hold = X_pre / sr if extra else t_hold
            if comp.kind == "const":
                return "ampeg_sustain=100"
            return f"ampeg_hold={hold:.5f} ampeg_decay={tau_to_sfz_time(comp.tau):.4f} ampeg_sustain=0"

        regions = []
        for j, comp in enumerate(comps):
            y = wavs[j]
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            gdb = 0.0
            if peak > 0.98:
                y *= 0.98 / peak
                gdb = 20 * np.log10(peak / 0.98)
            pos = (f"loop_start={loop_start} loop_end={loop_end}" if j == 0 else
                   f"delay={delay_s:.6f} ampeg_attack={X_pre / sr:.6f} loop_start={X_pre} loop_end={X_pre + N - 1}")
            regions.append(dict(kind=comp.kind, env=env_opcodes(comp, j > 0), gain_db=gdb, file=j, pos=pos))
        if noise is not None:
            y = wavs[-1]
            peak = float(np.max(np.abs(y))) if y.size else 0.0
            gdb = 0.0
            if peak > 0.98:
                y *= 0.98 / peak
                gdb = 20 * np.log10(peak / 0.98)
            # residual level envelope
            rc = model.resid_centers
            body_r = np.where((rc >= n0) & (rc <= seg.release_onset))[0]
            if body_r.size >= 3:
                gl = np.sqrt((model.resid_mag[0][:, body_r].astype(np.float64) ** 2).sum(axis=0) + 1e-20)
                gl = gl / (gl[:max(1, min(3, len(gl)))].mean() + 1e-20)
                ncomps, _, _ = fit_envelope_components(gl[None, None, :], (rc[body_r] - n0) / sr, klass, min(stages, 2))
                tot = sum(float(c.coef[0, 0]) for c in ncomps)
                noise_comps = [(c, float(c.coef[0, 0]) / tot) for c in ncomps if tot > 0 and c.coef[0, 0] / tot > 0.02]
            else:
                noise_comps = [(EnvComponent("const"), 1.0)]
            pos = f"delay={delay_s:.6f} ampeg_attack={X_pre / sr:.6f} loop_start={X_pre} loop_end={X_pre + N_noise - 1}"
            for comp, frac in noise_comps:
                regions.append(dict(kind="noise-" + comp.kind, env=env_opcodes(comp, True),
                                    gain_db=gdb + 20 * np.log10(max(frac, 1e-4)), file=len(wavs) - 1, pos=pos))
        key = int(round(dsp.hz_to_midi(self.f0_nom)))
        cents = 100 * (dsp.hz_to_midi(self.f0_nom) - key)
        wav_paths = []
        for j, y in enumerate(wavs):
            is_noise = noise is not None and j == len(wavs) - 1
            suffix = "" if j == 0 else ("_noise" if is_noise else f"_s{j + 1}")
            p = os.path.join(out_dir, f"{self.name}{tag}{suffix}.{cfg.out_format}")
            dsp.write_audio(p, y, sr, subtype="PCM_16")
            wav_paths.append(p)
        sfz_path = os.path.join(out_dir, f"{self.name}{tag}.sfz")
        with open(sfz_path, "w") as f:
            f.write(f"// sfzc laroche loop-locked oscillator-bank loop | q={self.res['q']:.2f} class={klass} "
                    f"f0={self.f0_nom:.2f}Hz ({cents:+.1f} cents from key {key}) B={model.B:.2e}\n")
            f.write(f"// loop {loop_start}..{loop_end} ({N} samples = {N/sr*1000:.1f} ms, {N*self.f0_nom/sr:.1f} periods), "
                    f"K={model.K}, noise loop {N_noise} samples, detune (energy-weighted) {detune_w:.2f} cents, "
                    f"max {np.abs(detune).max():.2f} cents\n")
            f.write(f"// envelope fit error {fit_err:.2f} dB (single-stage baseline {base_err:.2f} dB)"
                    + (f"; refinement loss {refine_info['loss_before']:.4f} -> {refine_info['loss_after']:.4f}"
                       if refine_info and refine_info.get("loss_before") is not None else "") + "\n")
            f.write("<global> amp_veltrack=0\n")
            for r in regions:
                f.write(f"<region> sample={os.path.basename(wav_paths[r['file']])} pitch_keycenter={key} lokey={key} hikey={key}"
                        f"  // {r['kind']}\n  loop_mode=loop_continuous {r['pos']} {xfade}\n"
                        f"  volume={r['gain_db']:.3f} {r['env']} ampeg_release={rel:.4f} {lfo}\n")
        size = sum(os.path.getsize(p) for p in wav_paths)
        return dict(sfz_path=sfz_path, wav_paths=wav_paths, key=key, cents=cents, loop_start=loop_start,
                    loop_end=loop_end, N=N, n0=n0, i0=i0, iN=N // model.hop, stages=len(comps), residual=residual,
                    total_s=sum(len(w) for w in wavs) / sr, nudge_cents_max=float(np.abs(detune).max()),
                    nudge_cents_w=detune_w, fit_err=fit_err, base_err=base_err, fileg=None, noise_rms=0.0, size=size,
                    duration_s=len(wavs[0]) / sr, release=rel, taus=[c.tau for c in comps], blend="laroche",
                    locked_freqs=f_lock.ravel().tolist(), orig_freqs=f_i.ravel().tolist(),
                    N_noise=N_noise, refine=refine_info, config=("laroche", stages, residual))

    # ---------------------------------------------------------------- driver
    def run(self, out_dir: str) -> LoopResult:
        os.makedirs(out_dir, exist_ok=True)
        self.analyse()
        if self.model is None:
            return self._write_oneshot(out_dir)
        res = self.res
        mod = self._modulation()
        moving = mod["vib_cents"] > 3.0 or mod["trem_db"] > 1.5
        frozen = self.cfg.frozen == "on" or (self.cfg.frozen == "auto" and not moving)
        self.log.append(f"modulation: vibrato {mod['vib_cents']:.1f} cents @ {mod['vib_rate']:.1f} Hz, tremolo {mod['trem_db']:.1f} dB "
                        f"-> {'frozen loop-locked partials' if frozen else 'tracked partials (parameter-domain closing)'}")
        if not frozen:
            # the report's advice for moving sustains: keep the movement; the tracked path does that
            self.cfg.hybrid = "off"
            r = super().run(out_dir)
            r.info["log"] = self.log + r.info.get("log", [])
            r.info["mode"] = "laroche-tracked"
            return r
        if self.cfg.max_total_s is not None and self.cfg.verify:
            configs = [(1, True), (2, True), (1, False)] if self.seg.klass == "sustain" else [(1, True), (2, True), (3, True)]
            configs = [(s, r) for s, r in configs if s <= max(res["stages"], 1)]
        else:
            configs = [(res["stages"], res["residual"])]
        n_try = max(1, self.cfg.n_candidates) if self.cfg.verify else 1
        if len(configs) > 1:
            n_try = max(1, min(n_try, 2))
        tried, best, ci = [], None, 0
        for stages_c, resid_c in configs:
            n_files = stages_c + (1 if resid_c else 0)
            cands = self._laroche_candidates(n_files, mod)
            for cost, i0, N in cands[:n_try]:
                tag = "" if ci == 0 else f"_c{ci}"
                ci += 1
                b = self.build_laroche(i0, N, stages_c, resid_c, out_dir, tag=tag)
                b["search_cost"] = cost
                if self.cfg.verify:
                    b["metric"] = self.evaluate(b["sfz_path"], b["key"], b["N"] / self.sr, b["loop_start"] / self.sr)
                    score = b["metric"]["score"]
                else:
                    b["metric"] = None
                    score = -cost
                tried.append(b)
                self.log.append(f"candidate {ci - 1} (laroche stages={stages_c} residual={resid_c}): N={b['N']} "
                                f"({b['N']/self.sr*1000:.1f} ms, {b['N']*self.f0_nom/self.sr:.0f} periods) cost={cost:.3f} "
                                f"score={score:.3f} detune={b['nudge_cents_w']:.2f}c total={b['total_s']:.3f}s")
                if best is None or score > best[0]:
                    best = (score, b)
        _, b = best
        canon_sfz = os.path.join(out_dir, f"{self.name}.sfz")
        if b["sfz_path"] != canon_sfz:
            canon = self.build_laroche(b["i0"], b["N"], b["config"][1], b["config"][2], out_dir, tag="")
            canon["metric"], canon["search_cost"] = b["metric"], b["search_cost"]
            b = canon
        for t in tried:
            if t["sfz_path"] != b["sfz_path"] and t["sfz_path"] != canon_sfz:
                for p in t["wav_paths"] + [t["sfz_path"]]:
                    if os.path.exists(p):
                        os.remove(p)
        # round-robin: a second loop set from a different sustain position
        if self.cfg.round_robin > 1:
            model = self.model
            i_alt = int(min(model.F - 2, b["i0"] + max(1, int(0.5 * (model.frame_at(self.seg.release_onset) - b["i0"])))))
            if model.centers[i_alt] + b["N"] < self.seg.release_onset:
                alt = self.build_laroche(i_alt, b["N"], b["config"][1], b["config"][2], out_dir, tag="_rr2")
                with open(b["sfz_path"]) as f:
                    main_txt = f.read()
                with open(alt["sfz_path"]) as f:
                    alt_txt = f.read()
                main_txt = main_txt.replace("<region>", "<region> seq_length=2 seq_position=1")
                alt_regions = "\n".join(l for l in alt_txt.splitlines() if not l.startswith("//") and not l.startswith("<global>"))
                alt_regions = alt_regions.replace("<region>", "<region> seq_length=2 seq_position=2")
                with open(b["sfz_path"], "w") as f:
                    f.write(main_txt + "\n// round-robin set 2\n" + alt_regions + "\n")
                os.remove(alt["sfz_path"])
                b["wav_paths"] += alt["wav_paths"]
                b["size"] += alt["size"]
                b["total_s"] += alt["total_s"]
                self.log.append(f"round-robin set 2 from {model.centers[i_alt]/self.sr:.2f}s")
        baseline_metric = None
        if self.cfg.baseline:
            bl = self.build_baseline(b["i0"], b["N"], out_dir)
            if self.cfg.verify:
                baseline_metric = self.evaluate(bl["sfz_path"], bl["key"], bl["N"] / self.sr, bl["loop_start"] / self.sr)
                self.log.append(f"baseline crossfade loop score={baseline_metric['score']:.3f}")
            b["baseline"] = bl
        # seam diagnostics on the emitted loop
        try:
            from .diagnostics import diagnose_loop

            w0, _ = dsp.load_audio(b["wav_paths"][0])
            diag = diagnose_loop(w0[b["loop_start"]: b["loop_end"] + 1], self.sr, np.array(b["locked_freqs"]),
                                 np.array(b["orig_freqs"]))
        except Exception as e:  # pragma: no cover
            diag = dict(error=repr(e))
        result = LoopResult(self.name, b["sfz_path"], b["wav_paths"], self.seg.klass, b["key"], float(b["cents"]),
                            self.f0_nom, float(self.model.B), b["loop_start"], b["loop_end"], b["N"],
                            b["n0"] / self.sr, b["stages"], b["residual"], self.model.K, b["duration_s"],
                            len(self.x) / self.sr, b["size"], total_audio_s=b["total_s"], metric=b["metric"],
                            baseline_metric=baseline_metric,
                            candidates=[dict(i0=t["i0"], N=t["N"], cost=t["search_cost"],
                                             score=(t["metric"] or {}).get("score")) for t in tried],
                            info=dict(log=self.log, mode="laroche", nudge_cents_max=b["nudge_cents_max"],
                                      nudge_cents_w=b["nudge_cents_w"], fit_err=b["fit_err"], taus=b["taus"],
                                      release=b["release"], N_noise=b["N_noise"], refine=b["refine"], diagnostics=diag,
                                      baseline=b.get("baseline", {}).get("sfz_path"), modulation=mod))
        with open(os.path.join(out_dir, f"{self.name}.json"), "w") as f:
            json.dump(asdict(result), f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
        return result
