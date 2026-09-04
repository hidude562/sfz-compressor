"""Checks for dctjoin.unaltered: the loop is never touched, the join is found where the recording
matches it, and the attack is what gets adapted.  (python3 -m pytest dctjoin/tests -q)"""
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from dctloop import loop_signal
from dctjoin.unaltered import find_join, fit_attack_level, release_time, replicate_unaltered

FS = 44100


def note(f0=440.0, attack=0.08, sustain=2.5, release=0.3, amps=(1.0, .5, .3, .2, .1), noise=0.002,
         stereo=True, swell_db=0.0, seed=0):
    """attack ramp -> (optional swell) -> steady -> exponential release, stereo with a 0.25 ms delay."""
    rng = np.random.default_rng(seed)
    n = int((attack + sustain + release) * FS)
    t = np.arange(n) / FS
    env = np.ones(n)
    na = int(attack * FS)
    env[:na] = (np.arange(na) / na) ** 2
    if swell_db:
        ns = int(0.3 * FS)
        env[na:na + ns] *= 10 ** (swell_db / 20 * (1 - np.arange(ns) / ns))
    nr = int(release * FS)
    env[-nr:] *= np.exp(-np.arange(nr) / FS / (release / 5))
    y = 0.3 * sum(a * np.cos(2 * np.pi * h * f0 * t + 0.3 * h) for h, a in enumerate(amps, 1)) * env
    y = y + noise * rng.standard_normal(n)
    if not stereo:
        return y[:, None]
    d = 11
    return np.stack([y, np.concatenate([np.zeros(d), y[:-d]]) + noise * rng.standard_normal(n)], 1)


def test_find_join_lands_where_the_recording_matches_the_loop():
    f0 = 440.0
    x = note(f0, noise=0.0)
    body = x[int(0.3 * FS): int(2.4 * FS)]
    loop, info = loop_signal(body, FS, 0.5, basis='dft', f0=f0)
    fj = find_join(x, loop, FS, f0, int(0.1 * FS), int(0.6 * FS))
    assert fj['ncc'] > 0.97, fj
    W = fj['W']
    J = fj['J']
    # the recording really does continue like the loop at J: correlation of the next 4 periods
    P4 = int(4 * FS / f0)
    a, b = x[J:J + P4, 0], loop[:P4, 0]
    assert np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.97
    assert abs(fj['level_db']) < 1.0


def test_find_join_prefers_a_level_matched_spot_when_waveforms_tie():
    """A purely periodic tone matches equally well once per period; the level term breaks the tie
    towards where the recording's level equals the loop's (i.e. after a swell has settled)."""
    f0 = 330.0
    x = note(f0, noise=0.0, swell_db=6.0)
    body = x[int(0.5 * FS): int(2.5 * FS)]
    loop, _ = loop_signal(body, FS, 0.5, basis='dft', f0=f0)
    fj = find_join(x, loop, FS, f0, int(0.09 * FS), int(0.6 * FS))
    assert fj['ncc'] > 0.95
    assert abs(fj['level_db']) < 1.5, fj          # not at the loud start of the swell (+6 dB)


def test_attack_level_ramp_arrives_at_the_loop_level_and_leaves_the_transient_alone():
    f0 = 440.0
    x = note(f0, noise=0.0)
    loop, _ = loop_signal(x[int(0.3 * FS): int(2.4 * FS)], FS, 0.5, basis='dft', f0=f0)
    loop = loop * 0.5                               # pretend the loop is 6 dB quieter
    onset, J = 0, int(0.4 * FS)
    x2, g, R = fit_attack_level(x, onset, J, loop, FS, f0)
    assert np.allclose(g, 0.5, atol=0.03), g
    W = int(4 * FS / f0)
    assert abs(np.sqrt(np.mean(x2[J - W:J] ** 2)) / np.sqrt(np.mean(loop[:W] ** 2)) - 1) < 0.05
    assert np.array_equal(x2[: J - R], x[: J - R])  # nothing before the ramp changed
    assert R <= (J - onset) // 2


def test_release_time_from_an_exponential_tail():
    x = note(440.0, release=0.6, noise=0.0)
    ro = int((0.08 + 2.5) * FS)
    T, D = release_time(x, FS, ro, len(x) - 1)
    # the tail falls 60 dB in (0.6/5) * ln(10^3) = 0.83 s; sfizz T = D * 78.2/60
    assert 0.6 < D < 1.1, D
    assert abs(T - D * 78.2 / 60) < 1e-6


@pytest.mark.parametrize('basis', ['dft', 'dct'])
def test_replicate_keeps_the_loop_byte_identical(basis):
    f0 = 466.16
    x = note(f0, sustain=3.0)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'horn-a#4.wav')
        sf.write(src, x, FS, subtype='PCM_24')
        r = replicate_unaltered(src, d, 0.5, basis=basis, sfizz=False, preview=False)
        assert r.loop_untouched and r.file_gain == 1.0
        assert r.loop_end - r.loop_start + 1 == r.L
        # and what dctloop alone would have made of the same file is the same loop
        from dctloop import find_body, loop_signal as ls_
        y, fs = sf.read(src, dtype='float64', always_2d=True)
        a, b = find_body(y, fs)
        ref, _ = ls_(y[a:b], fs, 0.5, basis=basis, f0=r.f0)
        out, _ = sf.read(r.outputs['audio'], dtype='float64', always_2d=True)
        got = out[r.loop_start: r.loop_end + 1]
        assert got.shape == ref.shape
        assert np.max(np.abs(got - ref)) < 2 / 2 ** 23          # 24-bit rounding only
        sfz = open(r.outputs['sfz']).read()
        assert f'loop_start={r.loop_start} loop_end={r.loop_end}' in sfz
        assert 'loop_mode=loop_continuous' in sfz
        if basis == 'dft':
            # the bridge lands the tail on the loop's entry: waveform match and per-harmonic continuity
            assert r.join_ncc > 0.9, r.join_ncc
            assert r.continuity['tail_ncc'] > 0.99, r.continuity
            assert r.continuity['excess_harm_step_db'] < 0.3, r.continuity
            assert r.continuity['excess_phase_err_deg'] < 5.0, r.continuity
            assert r.junction['dip_db'] > -1.0, r.junction
