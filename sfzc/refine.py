"""Stage-3 refinement: optimise per-partial and per-noise-band gains of an exactly periodic loop
against a multi-resolution log-magnitude STFT loss (spectral convergence + log L1) between the
tiled loop and the source sustain.  Frequencies stay grid-locked, so periodicity is preserved by
construction.  Needs PyTorch; falls back to unity gains if it is missing."""
from __future__ import annotations

import numpy as np


def split_noise_bands(noise: np.ndarray, sr: int, n_bands: int = 16, fmin: float = 60.0) -> np.ndarray:
    """Split a periodic noise loop (N,) into `n_bands` log-spaced bands via FFT masks -> (B, N)."""
    N = len(noise)
    Y = np.fft.rfft(noise)
    fb = np.fft.rfftfreq(N, 1 / sr)
    edges = np.geomspace(fmin, sr / 2, n_bands + 1)
    edges[0] = 0.0
    out = np.zeros((n_bands, N))
    for b in range(n_bands):
        m = (fb >= edges[b]) & (fb < edges[b + 1])
        out[b] = np.fft.irfft(Y * m, n=N)
    return out


def _avg_log_spectra(x, n_fft: int, hop: int):
    import torch

    w = torch.hann_window(n_fft, device=x.device)
    S = torch.stft(x, n_fft, hop_length=hop, window=w, return_complex=True).abs() ** 2  # (bins, frames)
    P = S.mean(dim=1)
    return P, torch.log(P + 1e-9)


def refine_gains(parts: np.ndarray, noise_bands: np.ndarray, target: np.ndarray, sr: int, iters: int = 80,
                 tile: int = 6, lr: float = 0.05, fft_sizes=(512, 1024, 2048), max_gain_db: float = 12.0,
                 max_noise_gain_db: float = 6.0) -> dict:
    """parts: (K, N) per-partial mono loops; noise_bands: (B, N); target: (n,) mono source sustain.

    Returns dict(part_gains (K,), noise_gains (B,), loss_before, loss_after)."""
    K, N = parts.shape
    B = noise_bands.shape[0] if noise_bands is not None and len(noise_bands) else 0
    try:
        import torch
    except Exception:  # pragma: no cover
        return dict(part_gains=np.ones(K), noise_gains=np.ones(B), loss_before=None, loss_after=None, skipped=True)
    dev = "cpu"
    P = torch.tensor(parts, dtype=torch.float32, device=dev)
    Nz = torch.tensor(noise_bands, dtype=torch.float32, device=dev) if B else None
    X = torch.tensor(np.asarray(target, np.float32), device=dev)
    g = torch.zeros(K, device=dev, requires_grad=True)
    h = torch.zeros(B, device=dev, requires_grad=True) if B else None
    lim = max_gain_db / 20 * np.log(10)
    lim_n = max_noise_gain_db / 20 * np.log(10)
    targets = {}
    for n_fft in fft_sizes:
        if len(X) >= n_fft:
            Pt, _ = _avg_log_spectra(X, n_fft, n_fft // 4)
            Pt = Pt / (Pt.sum() + 1e-12)  # spectral *shape*: the absolute level is set by the analysis
            targets[n_fft] = (Pt, torch.log(Pt + 1e-12))

    def loss_fn():
        y = torch.exp(g.clamp(-lim, lim)) @ P
        if B:
            y = y + torch.exp(h.clamp(-lim_n, lim_n)) @ Nz
        yt = y.repeat(tile)
        total = 0.0
        for n_fft, (Pt, Lt) in targets.items():
            Py, _ = _avg_log_spectra(yt, n_fft, n_fft // 4)
            Py = Py / (Py.sum() + 1e-12)
            Ly = torch.log(Py + 1e-12)
            sc = torch.norm(torch.sqrt(Py) - torch.sqrt(Pt)) / (torch.norm(torch.sqrt(Pt)) + 1e-9)
            total = total + sc + 0.1 * torch.mean(torch.abs(Ly - Lt))
        return total / max(len(targets), 1)

    params = [g] + ([h] if B else [])
    opt = torch.optim.Adam(params, lr=lr)
    with torch.no_grad():
        before = float(loss_fn())
    for _ in range(iters):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = float(loss_fn())
        pg = torch.exp(g.clamp(-lim, lim)).cpu().numpy()
        ng = torch.exp(h.clamp(-lim_n, lim_n)).cpu().numpy() if B else np.ones(0)
    return dict(part_gains=pg, noise_gains=ng, loss_before=before, loss_after=after, skipped=False)
