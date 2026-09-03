"""Quick self-tests on synthetic signals (run: python3 -m pytest tests/ or python3 tests/test_synthetic.py)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sfzc import dsp, harmonic, looper  # noqa: E402
from sfzc.envelope import fit_envelope_components  # noqa: E402
from sfzc.metric import evaluate_recreation  # noqa: E402

SR = 44100


def synth_note(sr=SR, dur=3.0, f0=220.0, vib_hz=5.0, vib_cents=12.0, noise=0.01, decay=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    f_inst = f0 * 2 ** (vib_cents / 1200 * np.sin(2 * np.pi * vib_hz * t))
    ph = 2 * np.pi * np.cumsum(f_inst) / sr
    x = np.zeros((len(t), 2))
    for k in range(1, 25):
        a = 0.4 / k
        for c, dph in ((0, 0.0), (1, 0.4 * k)):
            x[:, c] += a * np.cos(k * ph + dph)
    env = np.minimum(1.0, t / 0.05)  # attack
    if decay:
        env = env * np.exp(-t / decay)
    else:
        env = env * (1 - 0.2 * np.minimum(1, t / dur))
    env[-int(0.2 * sr):] *= np.linspace(1, 0, int(0.2 * sr))
    x *= env[:, None]
    x += noise * rng.standard_normal(x.shape) * env[:, None]
    return x.astype(np.float32)


def test_harmonic_analysis_accuracy():
    x = synth_note(vib_cents=0.0, noise=0.0)
    ft = (np.array([0, len(x)]), np.array([220.0, 220.0]))
    m = harmonic.analyze(x, SR, ft, 220.0, K_max=10, n_start=int(0.2 * SR), n_end=int(2.5 * SR))
    assert abs(m.freq[0, 0].mean() - 220.0) < 0.05
    assert np.allclose(m.amp[0, :3, 20] * 1.0, [0.4, 0.2, 0.4 / 3], rtol=0.03)
    assert m.harmonicity.mean() > 0.95


def test_envelope_factorisation():
    t = np.linspace(0, 4, 200)
    amp = np.zeros((1, 4, len(t)))
    for k in range(4):
        amp[0, k] = 0.5 / (k + 1) * np.exp(-t / 0.4) * (k + 1) / 2 + 0.2 / (k + 1) * np.exp(-t / 2.5)
    comps, err, base = fit_envelope_components(amp, t, "decay", 2)
    assert err < base and err < 1.0
    taus = sorted(c.tau for c in comps)
    assert 0.25 < taus[0] < 0.7 and 1.5 < taus[1] < 4.0


def test_loop_is_exactly_periodic_and_scores_well():
    x = synth_note()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tone.wav")
        dsp.write_audio(p, x, SR)
        r = looper.process_sample(p, d, looper.LoopConfig(q=0.6, n_candidates=1, verify=True, baseline=True))
        assert r.klass == "sustain"
        w, _ = dsp.load_audio(r.wav_paths[0])
        ls, le = r.loop_start, r.loop_end
        # wrap continuity: the step across the seam is not larger than typical steps
        step_seam = np.abs(w[ls] - w[le]).max()
        typical = np.percentile(np.abs(np.diff(w[ls:le], axis=0)), 99)
        assert step_seam < 1.5 * typical, (step_seam, typical)
        assert r.metric["score"] > 0.3, r.metric
        assert r.metric["seam_prominence_db"] < 6.0
        assert r.metric["score"] >= r.baseline_metric["score"] * 0.8


def test_oneshot_path():
    rng = np.random.default_rng(1)
    t = np.arange(int(0.4 * SR)) / SR
    x = (rng.standard_normal((len(t), 2)) * np.exp(-t / 0.05)[:, None] * 0.5).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "hit.wav")
        dsp.write_audio(p, x, SR)
        r = looper.process_sample(p, d, looper.LoopConfig(q=0.6, verify=False))
        assert r.klass == "oneshot"
        assert os.path.exists(r.sfz_path)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
