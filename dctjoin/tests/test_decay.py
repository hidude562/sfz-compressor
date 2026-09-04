"""dctjoin.decay: a decaying note comes back as attack + untouched loop + an SFZ envelope that
restores the original decay."""
import os
import tempfile

import numpy as np
import soundfile as sf

from dctjoin.decay import fit_decay, flatten, replicate_decaying, smooth_envelope

FS = 44100


def piano_like(f0=440.0, secs=4.0, a1=0.7, tau1=0.15, a2=0.3, tau2=2.5, amps=(1.0, .5, .3, .2, .12), seed=0):
    """Two-stage exponential decay on a harmonic tone with a 5 ms attack, stereo, a little noise."""
    rng = np.random.default_rng(seed)
    n = int(secs * FS)
    t = np.arange(n) / FS
    env = np.minimum(1.0, t / 0.005) * (a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2))
    y = sum(a * np.cos(2 * np.pi * h * f0 * t + 0.4 * h) for h, a in enumerate(amps, 1))
    x = 0.4 * env[:, None] * np.stack([y, np.roll(y, 9)], 1) + 0.0005 * rng.standard_normal((n, 2))
    return x


def test_flatten_makes_the_body_stationary():
    x = piano_like()
    J, end = int(0.5 * FS), int(3.0 * FS)
    flat, g = flatten(x, FS, J, end, J)
    pos, e = smooth_envelope(flat, FS)
    e_db = 20 * np.log10(e)
    assert e_db.max() - e_db.min() < 3.0, (e_db.max(), e_db.min())     # was ~25 dB of decay
    assert g[0] == pytest_approx(1.0, 0.1)


def pytest_approx(v, tol):
    class A:
        def __eq__(self, other):
            return abs(other - v) <= tol
    return A()


def test_fit_decay_recovers_two_stages():
    t = np.arange(0, 3.0, 0.005)
    g = 0.7 * np.exp(-t / 0.15) + 0.3 * np.exp(-t / 2.5)
    comps = fit_decay(t, g, n_max=3, w=1.0 / g ** 2)
    fit = sum(a * np.exp(-t / tau) for a, tau in comps)
    err = 20 * np.log10(fit / g)
    assert np.max(np.abs(err)) < 1.0, (comps, np.max(np.abs(err)))
    taus = sorted(tau for _, tau in comps)
    assert 0.1 < taus[0] < 0.25 and 1.5 < taus[-1] < 4.0, comps


def test_replicate_decaying_end_to_end():
    x = piano_like()
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'toy-a4.wav')
        sf.write(src, x, FS, subtype='PCM_24')
        r = replicate_decaying(src, d, 0.5, sfizz=True, preview=False)
        assert r.loop_untouched and r.attack_s <= 0.5 + 1e-6
        assert r.continuity['tail_ncc'] > 0.98, r.continuity
        assert r.envelope_fit_db[0] < 1.5, r.envelope_fit_db
        assert 2 <= len(r.envelope) <= 3
        # sfizz re-creates the decay: rendered envelope within 2 dB rms of the original over the note
        assert r.render_env_err_db is not None and r.render_env_err_db[0] < 2.0, r.render_env_err_db
        sfz = open(os.path.join(d, 'toy-a4.json')).read()
        assert '"hold_s"' in sfz
