"""dctloop.split — harmonic loop (a few periods) + long residual loop.

Why: a single loop of length L turns *everything* that is not a steady harmonic (noise, chorus,
vibrato sidebands) into a texture that repeats every L seconds; below ~1.5 s that repetition is
heard as a "swish" at 1/L Hz.  The harmonics themselves have period 1/f0 and never restart.  So:

* the grid lines at multiples of K (the h*f0 partials) become a **harmonic loop** of a few
  periods (a few hundred samples) with the original phases and inter-channel relations;
* everything else is re-interpolated onto a **residual loop** of its own, long, length (default
  3 s) with random phases — a new noise realisation with the analysed spectrum.  It is not tied
  to the sustain length because only its spectrum is taken from the recording.

The two play simultaneously as two SFZ regions; their loop lengths are incommensurate, so the
sum never repeats within a note.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import soundfile as sf
from scipy.fft import irfft

from .core import _rms, analyse_on_grid, fit_loop_length
from .metrics import seam_metrics, spectrum_match
from .pipeline import _fade, find_body, load_audio
from .pitch import estimate_f0, note_from_name, refine_f0


def fit_short_loop(fs: int, f0: float, max_periods: int = 32, tol_cents: float = 0.5,
                   periods: int | None = None) -> tuple[int, int, float]:
    """Shortest loop (any integer number of samples) holding an integer number K of periods of f0
    within tol_cents; otherwise the best fit up to max_periods.  Returns (L, K, cents error)."""
    P = fs / f0
    best = None
    for K in ([periods] if periods else range(1, max_periods + 1)):
        L = int(round(K * P))
        if L < 4:
            continue
        cents = 1200 * math.log2((K * fs / L) / f0)
        if periods is None and abs(cents) <= tol_cents:
            return L, K, cents
        cand = (abs(cents), K, L, cents)
        if best is None or cand < best:
            best = cand
    return best[2], best[1], best[3]


def split_loop(seg: np.ndarray, fs: int, f0: float, resid_seconds: float = 3.0,
               harm_periods: int | None = None, max_periods: int = 32,
               harm_bw_cents: float | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split the sustain ``seg`` (N x C) into (harmonic loop, residual loop, info).

    harm_bw_cents : None -> each harmonic takes only its strongest grid line (the peak-moved
                    main lobe; vibrato/chorus spread stays in the residual).  A value -> the
                    whole band +-bw cents around h*f0 is summed into the harmonic and removed
                    from the residual (steadier, drier partials).
    """
    if seg.ndim == 1:
        seg = seg[:, None]
    N, C = seg.shape
    # --- fine analysis grid: the longest loop the sustain allows, fitted to f0
    T = N / fs / 2 * 0.995
    L_a, K_a, _ = fit_loop_length(T, fs, f0)
    while N < 2 * L_a:
        T *= 0.98
        L_a, K_a, _ = fit_loop_length(T, fs, f0)
    q = N // L_a
    off = (N - q * L_a) // 2
    x = seg[off:off + q * L_a]
    a, ref, diag = analyse_on_grid(x, L_a)                 # (M_a+1, C)
    M_a = L_a // 2
    a[0] = 0.0
    a[M_a] = 0.0

    # --- harmonic lines
    H = int(0.45 * fs / f0)
    wfrac = 0.02 if harm_bw_cents is None else (2.0 ** (harm_bw_cents / 1200.0) - 1.0)
    A = np.zeros((H + 1, C))
    ph = np.zeros((H + 1, C))
    resid_a = a.copy()
    used = 0
    for h in range(1, H + 1):
        m0 = h * K_a
        if m0 >= M_a - 2:
            break
        w = max(2, int(round(m0 * wfrac)))
        lo, hi = max(1, m0 - w), min(M_a - 1, m0 + w)
        band = a[lo:hi + 1]
        for c in range(C):
            m = lo + int(np.argmax(band[:, c]))
            ph[h, c] = float(np.angle(ref[m, c]))
            if harm_bw_cents is None:
                A[h, c] = a[m, c]
                resid_a[m, c] = 0.0
            else:
                A[h, c] = float(np.sqrt(np.sum(band[:, c] ** 2)))
                resid_a[lo:hi + 1, c] = 0.0
        used = h

    # --- harmonic loop: a few exact periods, original phases
    L_h, K_h, cents = fit_short_loop(fs, f0, max_periods=max_periods, periods=harm_periods)
    Yh = np.zeros((L_h // 2 + 1, C), complex)
    for h in range(1, used + 1):
        b = h * K_h
        if b >= L_h // 2:
            break
        Yh[b] = A[h] * (L_h / 2.0) * np.exp(1j * ph[h])
    harm = irfft(Yh, n=L_h, axis=0)

    # --- residual: analysed power re-gridded onto its own loop length, random phases
    L_r = int(round(resid_seconds * fs))
    M_r = L_r // 2
    P_a = resid_a ** 2
    f_a = np.arange(M_a + 1) * fs / L_a
    f_r = np.arange(M_r + 1) * fs / L_r
    if L_r >= L_a:                       # finer grid: spread the power density
        P_r = np.stack([np.interp(f_r, f_a, P_a[:, c]) for c in range(C)], 1) * (L_a / L_r)
    else:                                # coarser grid: bin the lines
        P_r = np.zeros((M_r + 1, C))
        idx = np.clip(np.round(f_a * L_r / fs).astype(int), 0, M_r)
        np.add.at(P_r, idx, P_a)
    a_r = np.sqrt(P_r)
    a_r[0] = 0.0
    a_r[-1] = 0.0
    rng = np.random.default_rng(seed)
    phr = rng.uniform(-np.pi, np.pi, a_r.shape)
    resid = irfft(a_r * (L_r / 2.0) * np.exp(1j * phr), n=L_r, axis=0)

    e_h, e_r, e_x = np.mean(harm ** 2), np.mean(resid ** 2), np.mean(x ** 2)
    info = dict(N=int(q * L_a), analysis_offset=int(off), L_analysis=int(L_a), K_analysis=int(K_a),
                frames=diag.get('frames'), grid_offset=diag.get('grid_offset'),
                L_harm=int(L_h), K_harm=int(K_h), f0_harm=K_h * fs / L_h, harm_cents=float(cents),
                tune_cents=float(-cents), harmonics=int(used), harm_bw_cents=harm_bw_cents,
                L_resid=int(L_r), harm_energy_frac=float(e_h / (e_h + e_r + 1e-30)),
                energy_ratio_db=float(10 * np.log10((e_h + e_r + 1e-30) / (e_x + 1e-30))),
                harm_rms_db=float(20 * np.log10(np.sqrt(e_h) + 1e-30)),
                resid_rms_db=float(20 * np.log10(np.sqrt(e_r) + 1e-30)))
    return harm, resid, info


def write_sfz(path: str, harm_wav: str, resid_wav: str, key: int, tune_cents: float,
              L_h: int, L_r: int, release: float = 0.25) -> None:
    """Two regions on one key: the harmonic loop (with its pitch correction) and the residual."""
    with open(path, 'w') as fh:
        fh.write(f"// dctloop split: harmonic loop ({L_h} samples) + residual loop ({L_r} samples)\n"
                 f"<global> lokey={key} hikey={key} pitch_keycenter={key} amp_veltrack=0 "
                 f"loop_mode=loop_continuous ampeg_attack=0.005 ampeg_release={release}\n"
                 f"<region> sample={os.path.basename(harm_wav)} tune={tune_cents:.2f} loop_start=0 loop_end={L_h - 1}\n"
                 f"<region> sample={os.path.basename(resid_wav)} loop_start=0 loop_end={L_r - 1}\n")


def render_with_sfizz(sfz_path: str, key: int, seconds: float, fs: int) -> np.ndarray | None:
    """Render one held note with sfizz (unity gain) via the sibling sfzc package; None if unavailable."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from sfzc.render import render_sfz
        y = render_sfz(sfz_path, key, seconds, seconds + 0.5, sr=fs)
        return np.asarray(y, dtype=np.float64)
    except Exception as e:  # pragma: no cover - depends on the local sfizz build
        print(f'  (sfizz render skipped: {e})')
        return None


def _tile(x: np.ndarray, n: int) -> np.ndarray:
    return np.tile(x, (int(math.ceil(n / len(x))), 1))[:n]


@dataclass
class SplitResult:
    source: str
    fs: int
    f0: float
    key: int
    info: dict
    harm_seam: dict
    sum_spectrum: dict
    sfizz_check: dict
    outputs: dict


def process_split(path: str, out_dir: str, resid_seconds: float = 3.0, harm_periods: int | None = None,
                  harm_bw_cents: float | None = None, f0: float | None = None, start: float | None = None,
                  dur: float | None = None, stem: str | None = None, seed: int = 0,
                  verbose: bool = False, sfizz: bool = True) -> SplitResult:
    x, fs = load_audio(path)
    name = os.path.splitext(os.path.basename(path))[0]
    stem = stem or name
    if start is not None:
        a0 = int(start * fs)
        b0 = int((start + dur) * fs) if dur else len(x)
    else:
        a0, b0 = find_body(x, fs)
        if dur:
            mid = (a0 + b0) // 2
            a0, b0 = max(0, mid - int(dur * fs / 2)), min(len(x), mid + int(dur * fs / 2))
    seg = x[a0:b0]
    hint = f0 or note_from_name(name)
    if f0 is None:
        f0c = estimate_f0(seg, fs, hint)
        good = f0c[np.isfinite(f0c)]
        f0_coarse = float(np.exp(np.mean(np.log(good)))) if good.size else (hint or 0.0)
        f0c = refine_f0(seg, fs, f0_coarse)
        f0_used = float(np.exp(np.mean(np.log(f0c))))
    else:
        f0_used = float(f0)
    key = int(round(69 + 12 * math.log2(f0_used / 440.0)))

    harm, resid, info = split_loop(seg, fs, f0_used, resid_seconds=resid_seconds, harm_periods=harm_periods,
                                   harm_bw_cents=harm_bw_cents, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    p_h = os.path.join(out_dir, f'{stem}_harm.wav')
    p_r = os.path.join(out_dir, f'{stem}_resid.wav')
    p_sfz = os.path.join(out_dir, f'{stem}_split.sfz')
    for p, y in ((p_h, harm), (p_r, resid)):
        pk = float(np.max(np.abs(y)))
        sf.write(p, y * (0.999 / pk) if pk > 0.999 else y, fs, subtype='PCM_24')
    # the sampler applies the pitch correction, so the SFZ tune undoes the harmonic loop's rounding
    write_sfz(p_sfz, p_h, p_r, key, 1200 * math.log2(f0_used / info['f0_harm']), info['L_harm'], info['L_resid'])
    outputs = dict(harm=p_h, resid=p_r, sfz=p_sfz)

    seg_used = seg[info['analysis_offset']:info['analysis_offset'] + info['N']]
    n4 = int(4.0 * fs)
    tiled_sum = _tile(harm, n4) + _tile(resid, n4)
    sum_spec = spectrum_match(seg_used, fs, tiled_sum[:len(seg_used)] if len(tiled_sum) >= len(seg_used) else _tile(tiled_sum, len(seg_used)))
    harm_seam = seam_metrics(harm, fs) if info['L_harm'] >= 512 else {}

    check: dict = {}
    rendered = render_with_sfizz(p_sfz, key, 4.0, fs) if sfizz else None
    if rendered is not None:
        r = rendered[int(0.5 * fs):int(4.0 * fs)]
        t = tiled_sum[int(0.5 * fs):int(4.0 * fs)]
        sp = spectrum_match(t, fs, r)
        check = dict(rms_db=float(20 * np.log10(_rms(r) / _rms(t))), ltas_mean_abs_db=sp['ltas_mean_abs_db'],
                     ltas_max_abs_db=sp['ltas_max_abs_db'])
        p_sf = os.path.join(out_dir, f'{stem}_sfizz.wav')
        sf.write(p_sf, np.clip(rendered, -1, 1), fs, subtype='PCM_16')
        outputs['sfizz'] = p_sf

    # preview: original | harmonic alone | residual alone | both (sfizz render if we have it)
    gap = np.zeros((int(0.4 * fs), x.shape[1]))
    both = rendered[:n4] if rendered is not None else tiled_sum
    pv = np.concatenate([_fade(seg_used[:int(2 * fs)], fs), gap, _fade(_tile(harm, int(1.5 * fs)), fs), gap,
                         _fade(_tile(resid, int(1.5 * fs)), fs), gap, _fade(both, fs)], axis=0)
    pk = float(np.max(np.abs(pv)))
    if pk > 0.999:
        pv *= 0.999 / pk
    p_pv = os.path.join(out_dir, f'{stem}_split_preview.wav')
    sf.write(p_pv, pv, fs, subtype='PCM_16')
    outputs['preview'] = p_pv

    res = SplitResult(source=path, fs=fs, f0=f0_used, key=key, info=info, harm_seam=harm_seam,
                      sum_spectrum=sum_spec, sfizz_check=check, outputs=outputs)
    import json
    with open(os.path.join(out_dir, f'{stem}_split.json'), 'w') as fh:
        json.dump(asdict(res), fh, indent=1)
    if verbose:
        print(f'  [{name}] f0={f0_used:.2f}Hz key={key} harm={info["L_harm"]}smp ({info["K_harm"]} periods, '
              f'{info["harm_cents"]:+.2f}c -> tune {info["tune_cents"]:+.2f}) {info["harmonics"]} harmonics, '
              f'{info["harm_energy_frac"] * 100:.1f}% of energy; resid={info["L_resid"] / fs:.1f}s; '
              f'sum ltas {sum_spec["ltas_mean_abs_db"]:.2f}/{sum_spec["ltas_max_abs_db"]:.2f}dB'
              + (f'; sfizz vs tiled: {check["rms_db"]:+.2f}dB rms, {check["ltas_mean_abs_db"]:.2f}dB ltas' if check else ''))
    return res
