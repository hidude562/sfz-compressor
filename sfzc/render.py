"""Render an SFZ instrument with sfizz (via pysfizz) for verification."""
from __future__ import annotations

import os

import numpy as np


_GAIN_CACHE: dict[int, float] = {}


def renderer_gain_db(sr: int = 44100) -> float:
    """sfizz applies a fixed headroom gain (about -11.5 dB). Measure it once so renders are unity gain."""
    if sr in _GAIN_CACHE:
        return _GAIN_CACHE[sr]
    import tempfile
    import soundfile as sf
    import pysfizz

    with tempfile.TemporaryDirectory() as d:
        n = sr
        t = np.arange(n) / sr
        x = 0.5 * np.sin(2 * np.pi * 441.0 * t)
        wav = os.path.join(d, "cal.wav")
        sf.write(wav, np.stack([x, x], 1), sr, subtype="PCM_16")
        sfz = os.path.join(d, "cal.sfz")
        with open(sfz, "w") as f:
            f.write(f"<region> sample=cal.wav pitch_keycenter=69 lokey=69 hikey=69 amp_veltrack=0 "
                    f"loop_mode=loop_continuous loop_start=0 loop_end={n - 1} ampeg_sustain=100\n")
        synth = pysfizz.Synth(sample_rate=sr, block_size=256)
        synth.load_sfz_file(sfz, quiet=True)
        y = synth.render_note(69, 127, 1.0, 1.2)[0]
        seg = y[sr // 4: sr // 2]
        g = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) / np.sqrt(np.mean(x[sr // 4: sr // 2] ** 2)))
    _GAIN_CACHE[sr] = float(g)
    return float(g)


def render_sfz(sfz_path: str, key: int, note_on_s: float, render_s: float, sr: int = 44100,
               vel: int = 100, block_size: int = 256, unity_gain: bool = True) -> np.ndarray:
    """Render one note with sfizz. Returns float array (n, 2), compensated to unity gain."""
    import pysfizz

    synth = pysfizz.Synth(sample_rate=sr, block_size=block_size)
    if not synth.load_sfz_file(os.path.abspath(sfz_path), quiet=True):
        raise RuntimeError(f"sfizz could not load {sfz_path}")
    y = synth.render_note(int(key), int(vel), float(note_on_s), float(render_s))
    y = np.ascontiguousarray(y.T.astype(np.float32))
    if unity_gain:
        y *= 10 ** (-renderer_gain_db(sr) / 20)
    return y
