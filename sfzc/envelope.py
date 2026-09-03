"""Fitting the amplitude / brightness evolution of a note to SFZ envelope opcodes.

sfizz (and ARIA for the sustain=0 case) implement ampeg decay/release as a pure
exponential:  g(t) = exp(-9 t / T)  (i.e. -78.2 dB per T seconds), with the
sustain level acting as a floor.  We therefore never rely on the sustain level:
every envelope is expressed as a sum of {constant, exp(-t/tau)} components,
one SFZ region per component, all regions playing phase-identical loops so the
sum is coherent.  fileg follows the same exponential rule (in cents).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import lpf2_mag2

SFZ_EXP_CONST = 9.0  # sfizz: rate = exp(-9 / (T * sr)) per sample


def tau_to_sfz_time(tau: float) -> float:
    return float(np.clip(SFZ_EXP_CONST * tau, 0.001, 100.0))


@dataclass
class EnvComponent:
    kind: str                 # 'const' | 'exp'
    tau: float = 0.0          # seconds (for 'exp')
    coef: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (C, K) signed amplitudes

    def value(self, t: np.ndarray) -> np.ndarray:
        if self.kind == "const":
            return np.ones_like(t, dtype=float)
        return np.exp(-np.maximum(t, 0.0) / self.tau)


def _weighted_ls(basis: np.ndarray, y: np.ndarray, w: np.ndarray, nonneg: bool) -> np.ndarray:
    """Solve min || sqrt(w) (basis @ c - y) || for c (shape (n_basis,)), optionally c >= 0."""
    sw = np.sqrt(w)[:, None]
    A = basis * sw
    b = y * sw[:, 0]
    if nonneg:
        from scipy.optimize import nnls

        c, _ = nnls(A, b)
        return c
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c


def fit_envelope_components(amp: np.ndarray, t: np.ndarray, klass: str, n_stages: int,
                            weights: np.ndarray | None = None,
                            sustain_floor: float = 0.4) -> tuple[list[EnvComponent], float, float]:
    """Factorise amp[c, k, i] ~ sum_j coef_j[c, k] * e_j(t_i).

    Returns (components, fit_error_db, baseline_error_db).  For the 'decay' class the
    components are exponentials with non-negative coefficients; for 'sustain' the first
    component is a constant and an optional signed exponential captures a slow drift.
    """
    C, K, F = amp.shape
    if weights is None:
        weights = np.ones(F)
    # per-partial importance: mean power (mid) with a soft floor
    pk = (amp.mean(axis=0) ** 2).mean(axis=1)
    pk = pk / (pk.max() + 1e-20)
    pk = np.maximum(pk, 1e-4)

    # audibility weight of every (partial, frame): power fraction within the frame, so that the
    # error is measured where the partial is heard (decayed partials do not steer the fit)
    pw = amp.mean(axis=0) ** 2
    frac = pw / (pw.sum(axis=0, keepdims=True) + 1e-20)
    wkt = np.sqrt(frac) * weights[None, :]

    Y = amp.reshape(C * K, F)                       # (P, F)
    Wf = np.tile(wkt + 1e-6, (C, 1))                # (P, F) fit weights
    Werr = np.tile(wkt, (C, 1))
    valid = Y.max(axis=1) > 0
    nonneg = klass == "decay"

    def fit_with(basis_fns):
        """Batched weighted least squares over all partials (non-negative by active-set enumeration)."""
        B = np.stack([f(t) for f in basis_fns], axis=1)  # (F, J)
        J = B.shape[1]
        subsets = [tuple(range(J))] if not nonneg else [s for r in range(J, 0, -1)
                                                          for s in __import__("itertools").combinations(range(J), r)]
        best_c = np.zeros((Y.shape[0], J))
        best_r = np.full(Y.shape[0], np.inf)
        for s in subsets:
            Bs = B[:, s]                                             # (F, j)
            A = np.einsum("fj,pf,fl->pjl", Bs, Wf, Bs)               # (P, j, j)
            A += 1e-10 * np.trace(A, axis1=1, axis2=2)[:, None, None] * np.eye(len(s))[None] + 1e-18
            b = np.einsum("fj,pf,pf->pj", Bs, Wf, Y)                 # (P, j)
            cs = np.linalg.solve(A, b[..., None])[..., 0]            # (P, j)
            feas = (cs >= 0).all(axis=1) if nonneg else np.ones(Y.shape[0], bool)
            pred = cs @ Bs.T                                         # (P, F)
            r = np.sum(Wf * (pred - Y) ** 2, axis=1)
            take = feas & (r < best_r)
            best_r[take] = r[take]
            full = np.zeros((Y.shape[0], J))
            full[:, list(s)] = cs
            best_c[take] = full[take]
        pred = best_c @ B.T
        e = 20 * np.log10(np.maximum(pred, 1e-9) / (Y + 1e-9))
        e[~valid] = 0.0
        err = float(np.sum(Werr[valid] * e[valid] ** 2) / max(np.sum(Werr[valid]), 1e-12))
        coefs = best_c.T.reshape(J, C, K)
        coefs[:, ~valid.reshape(C, K)] = 0.0
        return coefs, err

    grid = np.geomspace(0.05, 30.0, 28)
    if klass == "decay":
        exp_fn = lambda tau: (lambda tt, tau=tau: np.exp(-tt / tau))  # noqa: E731

        def fit_taus(taus):
            coefs, err = fit_with([exp_fn(tau) for tau in taus])
            return err, [EnvComponent("exp", tau, coefs[j]) for j, tau in enumerate(taus)]

        if n_stages <= 1:
            best = min((fit_taus([tau]) for tau in grid), key=lambda r: r[0])
        else:
            # 2 stages: exhaustive over a coarse grid, then coordinate descent on the fine grid,
            # adding a 3rd (or more) stage greedily
            coarse = grid[::2]
            best = None
            for a, tau1 in enumerate(coarse):
                for tau2 in coarse[a + 2:]:
                    r = fit_taus([tau1, tau2])
                    if best is None or r[0] < best[0]:
                        best = r
            taus = [c.tau for c in best[1]]
            for _ in range(n_stages - 2):
                cand = min((fit_taus(sorted(taus + [tau])) for tau in grid if all(abs(np.log(tau / x)) > 0.3 for x in taus)),
                           key=lambda r: r[0], default=None)
                if cand is None or cand[0] > 0.97 * best[0]:
                    break
                best = cand
                taus = [c.tau for c in best[1]]
            for _round in range(1):
                for j in range(len(taus)):
                    for tau in grid:
                        trial = list(taus)
                        trial[j] = tau
                        if len(set(np.round(np.log(trial), 1))) < len(trial):
                            continue
                        r = fit_taus(sorted(trial))
                        if r[0] < best[0]:
                            best = r
                            taus = [c.tau for c in best[1]]
        err, comps = best
        base_err = fit_taus([grid[len(grid) // 2]])[0]
        return comps, float(np.sqrt(err)), float(np.sqrt(base_err))
    # sustain
    coefs0, err0 = fit_with([lambda tt: np.ones_like(tt)])
    comps = [EnvComponent("const", 0.0, coefs0[0])]
    if n_stages >= 2:
        best = None
        for tau in grid:
            coefs, err = fit_with([lambda tt: np.ones_like(tt), lambda tt, tau=tau: np.exp(-tt / tau)])
            if best is None or err < best[0]:
                best = (err, tau, coefs)
        err, tau, coefs = best
        if err < 0.85 * err0:
            c2 = coefs[1]
            # keep the dominant sign only (one region can only carry one polarity per WAV)
            pos = np.sum(np.maximum(c2, 0) ** 2)
            neg = np.sum(np.maximum(-c2, 0) ** 2)
            c2 = np.maximum(c2, 0) if pos >= neg else np.minimum(c2, 0)
            c1 = coefs[0].copy()
            # a held note must not fade to silence: keep the constant part >= sustain_floor of the
            # level at the loop start for decaying partials
            s = c1 + c2
            low = (c2 > 0) & (c1 < sustain_floor * s)
            c1[low] = sustain_floor * s[low]
            c2[low] = (1 - sustain_floor) * s[low]
            comps = [EnvComponent("const", 0.0, c1), EnvComponent("exp", tau, c2)]
            return comps, float(np.sqrt(err)), float(np.sqrt(err0))
    return comps, float(np.sqrt(err0)), float(np.sqrt(err0))


def make_exact_at_start(comps: list[EnvComponent], amp0: np.ndarray) -> None:
    """Rescale coefficients so that sum_j coef_j == amp0 (C, K) at t = 0 (loop start)."""
    total = sum(c.coef for c in comps)
    r = np.ones_like(amp0)
    ok = (np.abs(total) > 1e-9) & (amp0 > 0)
    r[ok] = amp0[ok] / total[ok]
    r = np.clip(r, 0.5, 2.0)
    for c in comps:
        c.coef = c.coef * r


@dataclass
class FilEG:
    cutoff: float
    depth_cents: float
    decay_time: float

    def opcodes(self, hold: float) -> str:
        return (f"fil_type=lpf_2p cutoff={self.cutoff:.1f} fileg_depth={self.depth_cents:.0f} "
                f"fileg_hold={hold:.4f} fileg_decay={self.decay_time:.4f} fileg_sustain=0")


def fit_fileg(loop_amp: np.ndarray, loop_freq: np.ndarray, target_centroid: np.ndarray, t: np.ndarray,
              min_drop: float = 0.08) -> FilEG | None:
    """Fit a darkening low-pass envelope so that the loop's harmonic centroid follows target_centroid(t).

    loop_amp: (K,) mid amplitudes of the loop, loop_freq: (K,) Hz.
    """
    p = loop_amp ** 2
    if p.sum() <= 0:
        return None
    c_loop = float(np.sum(p * loop_freq) / p.sum())
    tc = np.asarray(target_centroid, float)
    good = np.isfinite(tc) & (tc > 0)
    if good.sum() < 4 or np.min(tc[good]) > c_loop * (1 - min_drop):
        return None

    def centroid_for(fc):
        g = p * lpf2_mag2(loop_freq, fc)
        return np.sum(g * loop_freq) / (g.sum() + 1e-20)

    fcs = np.full(len(tc), np.nan)
    for i in np.where(good)[0]:
        target = min(tc[i], c_loop * 0.999)
        lo, hi = np.log(40.0), np.log(22000.0)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if centroid_for(np.exp(mid)) > target:
                hi = mid
            else:
                lo = mid
        fcs[i] = np.exp(0.5 * (lo + hi))
    ok = np.isfinite(fcs)
    fc_end = float(np.percentile(fcs[ok], 5))
    fc_end = max(fc_end, 40.0)
    cents = 1200 * np.log2(np.maximum(fcs[ok], fc_end) / fc_end)
    tt = t[ok]
    sel = cents > 30
    if sel.sum() < 3:
        return None
    # log(cents) = log(depth) - 9 t / T   (weighted towards early frames where it matters most)
    A = np.vstack([np.ones(sel.sum()), -tt[sel]]).T
    wgt = np.sqrt(np.maximum(cents[sel], 1.0))
    sol, *_ = np.linalg.lstsq(A * wgt[:, None], np.log(cents[sel]) * wgt, rcond=None)
    depth = float(np.exp(sol[0]))
    rate = float(sol[1])
    if rate <= 0:
        return None
    T = SFZ_EXP_CONST / rate
    depth = float(np.clip(depth, 0, 12000))
    return FilEG(cutoff=fc_end, depth_cents=depth, decay_time=float(np.clip(T, 0.01, 100)))


def fit_release_time(env_t: np.ndarray, env_db: np.ndarray, sr: int, n_from: int, n_to: int,
                     default: float = 0.4) -> float:
    """Release time (sfizz convention) from the terminal decay slope of the recorded note."""
    sel = (env_t >= n_from) & (env_t <= n_to)
    if sel.sum() < 4:
        return default
    tt = env_t[sel] / sr
    e = env_db[sel]
    top = e[0]
    keep = e > top - 45
    if keep.sum() < 4:
        return default
    A = np.vstack([tt[keep], np.ones(keep.sum())]).T
    slope, _ = np.linalg.lstsq(A, e[keep], rcond=None)[0]
    if slope >= -5:
        return default
    return float(np.clip(78.17 / -slope, 0.03, 8.0))
