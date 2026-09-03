"""Synthetic checks for dctloop (run: python3 -m pytest dctloop/tests -q)."""
import math

import numpy as np
import pytest

from dctloop import (fit_loop_length, harmonic_am, loop_signal, make_loop, note_from_name, refine_f0,
                     seam_metrics)


FS = 44100


def tone(f0, secs, phases=None, amps=(1.0, 0.5, 0.25, 0.125), fs=FS, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(int(secs * fs))
    y = np.zeros(len(n))
    for h, a in enumerate(amps, 1):
        ph = phases[h - 1] if phases is not None else 0.0
        y += a * np.cos(2 * np.pi * h * f0 * n / fs + ph)
    return y + noise * rng.standard_normal(len(n))


def test_note_parsing():
    assert abs(note_from_name('trumpet-a#4') - 466.1637) < 1e-3
    assert abs(note_from_name('1st-violins-sus-a4.wav') - 440.0) < 1e-9
    assert abs(note_from_name('cello-c3') - 130.8128) < 1e-3
    assert note_from_name('2nd-violins-piz-rr2-e4') == pytest.approx(329.6276, abs=1e-3)
    assert note_from_name('snare-hit') is None


def test_fit_loop_length_exact_periods():
    L, K, cents = fit_loop_length(1.5, FS, 440.0)
    assert L % 2 == 0 and abs(K * FS / L - 440.0) / 440.0 < 1e-4
    assert abs(cents) < 0.2
    # 22 periods of 440 Hz at 44.1 kHz are exactly 2205 samples -> even rounding is 2204/2206
    L, K, _ = fit_loop_length(0.05, FS, 440.0)
    assert abs(L - K * FS / 440.0) <= 1.0


@pytest.mark.parametrize('basis', ['dct', 'dft'])
def test_loop_is_exactly_periodic_and_sounds_like_the_input(basis):
    f0 = 220.0
    rng = np.random.default_rng(1)
    phases = rng.uniform(-np.pi, np.pi, 4)
    x = tone(f0, 4.0, phases)[:, None]
    L, K, _ = fit_loop_length(1.0, FS, f0)
    loop, info = make_loop(x, FS, L, basis=basis)
    assert len(loop) == L and info['q'] == 4
    # exact periodicity: the seam is no bigger than any other step in the signal
    step = np.abs(np.diff(np.concatenate([loop, loop[:1]]), axis=0))
    assert step[-1, 0] <= np.percentile(step[:, 0], 99.5)
    # harmonic amplitudes are recovered (spectrum on the loop grid)
    Y = np.abs(np.fft.rfft(loop[:, 0])) * 2 / L
    amps = [Y[h * K] for h in range(1, 5)]
    assert np.allclose(amps, [1, .5, .25, .125], rtol=0.15, atol=0.02), amps
    # nothing between the harmonics
    off = np.delete(Y, [h * K for h in range(1, 5)])
    assert off.max() < 0.03


def test_dct_loop_is_a_palindrome():
    x = tone(330.0, 3.0, noise=0.05)[:, None]
    L, _, _ = fit_loop_length(0.5, FS, 330.0)
    loop, _ = make_loop(x, FS, L, basis='dct')
    assert np.allclose(loop, loop[::-1])
    loop2, _ = make_loop(x, FS, L, basis='dft')
    assert not np.allclose(loop2, loop2[::-1])


def test_snap_amplitude_is_phase_independent():
    """The DCT sees cos(phi) in one bin and sin(phi) in the neighbours: band energy must not
    depend on the phase of the partial."""
    f0 = 500.0
    L, K, _ = fit_loop_length(0.4, FS, f0)
    got = []
    for ph in np.linspace(0, np.pi, 9):
        x = tone(f0, 2.0, [ph, ph, ph, ph], amps=(1.0,))[:, None]
        loop, _ = make_loop(x, FS, L, basis='dct')
        got.append(np.abs(np.fft.rfft(loop[:, 0]))[K] * 2 / L)
    got = np.array(got)
    assert got.max() / got.min() < 1.05, got


def test_calibration_is_unity_on_noise_and_tones():
    """White noise comes back at unity gain and a tone with noise keeps both levels."""
    rng = np.random.default_rng(3)
    for q in (2, 4):
        x = rng.standard_normal((q * 22050, 1))
        for basis in ('dct', 'dft'):
            _, info = make_loop(x, FS, 22050, basis=basis)
            assert abs(info['gain'] - 1.0) < 0.05, (q, basis, info['gain'])
    n = np.arange(4 * FS)
    x = (np.cos(2 * np.pi * 220 * n / FS + 0.7) + 0.05 * rng.standard_normal(len(n)))[:, None]
    loop, info = make_loop(x, FS, 22050, basis='dct')
    Y = np.abs(np.fft.rfft(loop[:, 0])) * 2 / len(loop)
    K = int(round(220 * 22050 / FS))
    assert abs(Y[K] - 1.0) < 0.02
    assert abs(np.sqrt(np.sum(np.delete(Y, K) ** 2) / 2) - 0.05) < 0.01


def test_seam_metric_flags_a_bad_join():
    f0 = 261.6
    x = tone(f0, 4.0, [0.3, 1.1, 2.0, -1.0])[:, None]
    L, K, _ = fit_loop_length(1.0, FS, f0)
    loop, _ = make_loop(x, FS, L, basis='dct')
    good = seam_metrics(loop, FS)
    bad = seam_metrics(x[:L + 37], FS)  # a raw cut, not a whole number of periods
    assert good['seam_flux_ratio'] < 2.0
    assert bad['seam_flux_ratio'] > 2 * good['seam_flux_ratio']


def test_needs_two_loop_lengths():
    x = tone(440.0, 0.5)[:, None]
    with pytest.raises(ValueError):
        make_loop(x, FS, 2 * (int(0.4 * FS) // 2))


def test_off_grid_partial_is_moved_not_split():
    """A partial exactly between two grid bins must become ONE component (moved by half a bin),
    not two equal ones beating at 1/L Hz."""
    L = 22050
    f = (300 + 0.5) * FS / L                      # half-way between grid bins 300 and 301
    n = np.arange(4 * FS)
    x = np.cos(2 * np.pi * f * n / FS + 0.4)[:, None]
    for basis in ('dct', 'dft'):
        loop, _ = make_loop(x, FS, L, basis=basis)
        Y = np.abs(np.fft.rfft(loop[:, 0])) * 2 / L
        big, small = max(Y[300], Y[301]), min(Y[300], Y[301])
        assert big > 0.9 and small < 0.1, (basis, Y[299:303])


def test_refine_f0_is_precise():
    f_true = 441.73
    n = np.arange(3 * FS)
    x = (np.cos(2 * np.pi * f_true * n / FS) + 0.3 * np.cos(2 * np.pi * 2 * f_true * n / FS + 1))[:, None]
    f = refine_f0(x, FS, 440.0)                    # pyin-style coarse start, 7 cents off
    assert abs(f[0] - f_true) < 0.02


def test_stereo_signs_keep_the_image():
    """DCT signs are chosen relative to channel 0 so a coherent stereo pair stays coherent."""
    rng = np.random.default_rng(5)
    f0 = 233.1
    L, K, _ = fit_loop_length(1.0, FS, f0)
    n = np.arange(4 * FS)
    left = sum(a * np.cos(2 * np.pi * h * f0 * n / FS + p) for h, (a, p) in enumerate(zip([1, .6, .3, .2], rng.uniform(-3, 3, 4)), 1))
    delay = 13                                                 # right = left delayed 0.3 ms + own noise
    right = np.concatenate([np.zeros(delay), left[:-delay]])
    x = np.stack([left + 0.05 * rng.standard_normal(len(n)), right + 0.05 * rng.standard_normal(len(n))], 1)
    loop, _ = make_loop(x, FS, L, basis='dct')
    c_orig = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    c_loop = np.corrcoef(loop[:, 0], loop[:, 1])[0, 1]
    assert c_loop > 0.5 and abs(c_loop - c_orig) < 0.3, (c_orig, c_loop)


def test_split_separates_harmonics_from_noise():
    from dctloop.split import fit_short_loop, split_loop
    rng = np.random.default_rng(7)
    f0 = 311.13
    n = np.arange(int(3.5 * FS))
    amps = [1.0, 0.5, 0.25, 0.125]
    phases = rng.uniform(-np.pi, np.pi, 4)
    tone_part = sum(a * np.cos(2 * np.pi * h * f0 * n / FS + p) for h, (a, p) in enumerate(zip(amps, phases), 1))
    noise = 0.1 * rng.standard_normal(len(n))
    x = (tone_part + noise)[:, None]
    harm, resid, info = split_loop(x, FS, f0, resid_seconds=2.0)
    # harmonic loop: exact periods, right amplitudes, essentially no noise
    L, K = info['L_harm'], info['K_harm']
    assert abs(info['harm_cents']) <= 0.5 and len(harm) == L
    Y = np.abs(np.fft.rfft(harm[:, 0])) * 2 / L
    assert np.allclose([Y[h * K] for h in range(1, 5)], amps, rtol=0.1, atol=0.02)
    assert np.sqrt(np.mean(harm ** 2)) == pytest.approx(np.sqrt(np.mean(tone_part ** 2)), rel=0.1)
    # residual: the noise level, and nothing left at the harmonics
    assert np.sqrt(np.mean(resid ** 2)) == pytest.approx(0.1, rel=0.2)
    R = np.abs(np.fft.rfft(resid[:, 0])) * 2 / len(resid)
    Lr = len(resid)
    for h in range(1, 5):
        b = int(round(h * f0 * Lr / FS))
        assert R[b - 1:b + 2].max() < 0.03
    assert len(resid) == int(2.0 * FS)
    # short-loop fit
    L2, K2, c2 = fit_short_loop(FS, 440.0)
    assert abs(c2) <= 0.5 and L2 == int(round(K2 * FS / 440.0))


def test_harmonic_lock_removes_loop_rate_wah():
    """A tone whose pitch drifts by a few cents during the analysis spreads each harmonic over
    neighbouring grid bins; those beat with the harmonic once per loop.  Locking the harmonics
    to h*K must leave the neighbours far down."""
    f0 = 440.0
    n = np.arange(4 * FS)
    drift = 1 + 0.003 * np.sin(2 * np.pi * 0.2 * n / FS)              # +-5 cents, slow
    ph = 2 * np.pi * np.cumsum(f0 * drift) / FS
    x = sum(a * np.cos(h * ph) for h, a in enumerate([1, .5, .4, .3, .2], 1))[:, None]
    L, K, _ = fit_loop_length(0.25, FS, f0)
    free, _ = make_loop(x, FS, L, basis='dct')
    locked, _ = make_loop(x, FS, L, basis='dct', lock=f0)
    wah_free = harmonic_am(free, FS, f0, nharm=5)['max_harmonic_am_db']
    wah_locked = harmonic_am(locked, FS, f0, nharm=5)['max_harmonic_am_db']
    assert wah_free > 3.0, wah_free
    assert wah_locked < 0.5, wah_locked
    Y = np.abs(np.fft.rfft(locked[:, 0])) * 2 / L
    assert np.allclose([Y[h * K] for h in range(1, 6)], [1, .5, .4, .3, .2], rtol=0.1)


def test_loop_signal_end_to_end():
    """The array-level entry point: f0 estimated, loop fitted to whole periods, both bases."""
    f0 = 311.13
    n = np.arange(3 * FS)
    x = sum(a * np.cos(2 * np.pi * h * f0 * n / FS + h) for h, a in enumerate([1, .5, .3], 1))
    x = np.stack([x, np.roll(x, 7)], 1) + 0.01 * np.random.default_rng(0).standard_normal((len(n), 2))
    for basis in ('dct', 'dft'):
        loop, info = loop_signal(x, FS, 0.5, basis=basis, hint=310.0)
        assert abs(info['f0_used'] - f0) < 0.05, info['f0_used']
        assert info['L'] == len(loop) and abs(info['seconds'] - 0.5) < 0.01
        assert abs(info['K'] * FS / info['L'] - f0) / f0 < 1e-4
        assert harmonic_am(loop, FS, info['f0'], nharm=3)['max_harmonic_am_db'] < 0.5
    # too short an input shortens the loop instead of failing
    loop, info = loop_signal(x[:int(0.7 * FS)], FS, 0.5, f0=f0)
    assert info['shortened'] and len(loop) <= int(0.35 * FS)
