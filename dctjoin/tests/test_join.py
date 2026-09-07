"""Synthetic checks for dctjoin (run from sfz_compressor: python3 -m pytest dctjoin/tests -q)."""
import json
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from dctloop import fit_loop_length, loop_signal, make_loop
from dctjoin import (assign_key_ranges, join, junction_metrics, key_and_tune, level_match, morph_tail,
                     phase_align_loop, replicate_file, rotate_loop, segment_note, splice)

FS = 44100


def synth_note(f0=440.0, attack=0.08, sustain=2.5, release=0.3, amps=(1.0, .5, .3, .2, .1),
               detune=None, phases=None, noise=0.002, stereo=False, seed=0, fs=FS):
    """attack ramp -> steady -> exponential release; ``detune`` = per-harmonic multipliers."""
    rng = np.random.default_rng(seed)
    n = int((attack + sustain + release) * fs)
    t = np.arange(n) / fs
    env = np.ones(n)
    na = int(attack * fs)
    env[:na] = (np.arange(na) / na) ** 2
    nr = int(release * fs)
    env[-nr:] *= np.exp(-np.arange(nr) / fs / (release / 5))
    y = np.zeros(n)
    for h, a in enumerate(amps, 1):
        mult = detune[h - 1] if detune else 1.0
        ph = phases[h - 1] if phases else 0.3 * h
        y += a * np.cos(2 * np.pi * h * f0 * mult * t + ph)
    y = env * y + noise * rng.standard_normal(n)
    if stereo:
        d = 11
        return np.stack([y, np.concatenate([np.zeros(d), y[:-d]]) + noise * rng.standard_normal(n)], 1)
    return y[:, None]


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------------------- segmentation

def test_segment_finds_attack_end_and_release():
    x = synth_note(attack=0.08, sustain=2.5, release=0.3)
    s = segment_note(x, FS)
    assert s.onset < int(0.02 * FS)
    ae = (s.attack_end - s.onset) / FS
    assert 0.08 <= ae <= 0.3, ae
    ro = s.release_onset / FS
    assert abs(ro - (0.08 + 2.5)) < 0.2, ro
    assert s.release_onset > s.attack_end


# ---------------------------------------------------------------------------- alignment

def _loop_from(x, J, f0, seconds=0.25, basis='dft'):
    loop, info = loop_signal(x[J:], FS, seconds, basis=basis, f0=f0)
    return loop, info


def test_phase_align_makes_the_loop_continuous_with_the_recording():
    """With an inharmonic partial the inter-partial phases drift, so a rotation cannot match all
    of them at once; taking the recording's phases at the join can."""
    f0 = 440.0
    x = synth_note(f0, detune=(1.0, 1.0, 1.0023, 1.0, 1.0), noise=0.0)
    J = int(0.4 * FS)
    loop, info = _loop_from(x, J, f0)
    P4 = int(4 * FS / f0)
    c_raw = corr(loop[:P4], x[J:J + P4])
    rot, _ = rotate_loop(loop, x, J, FS)
    c_rot = corr(rot[:P4], x[J:J + P4])
    ali = phase_align_loop(loop, x, J)
    c_ali = corr(ali[:P4], x[J:J + P4])
    assert c_ali > 0.97, (c_raw, c_rot, c_ali)
    assert c_ali >= c_rot - 1e-3, (c_rot, c_ali)
    # magnitudes are untouched
    assert np.allclose(np.abs(np.fft.rfft(ali[:, 0])), np.abs(np.fft.rfft(loop[:, 0])), rtol=1e-6, atol=1e-9)


def test_rotation_aligns_a_purely_harmonic_tone():
    f0 = 330.0
    x = synth_note(f0, noise=0.0)
    J = int(0.5 * FS)
    loop, _ = _loop_from(x, J, f0)
    rot, tau = rotate_loop(loop, x, J, FS)
    P4 = int(4 * FS / f0)
    assert corr(rot[:P4], x[J:J + P4]) > 0.97
    assert 0 <= tau < len(loop)


def test_phase_align_needs_material_on_both_sides_of_the_join():
    x = synth_note(440.0, sustain=0.6)
    L = 2 * (int(0.25 * FS) // 2)
    loop = np.zeros((L, 1))
    with pytest.raises(ValueError):
        phase_align_loop(loop, x, len(x) - 10)
    with pytest.raises(ValueError):
        phase_align_loop(loop, x, 10)


def test_phase_align_is_exact_for_an_off_grid_partial():
    """The zero-phase frame is the point: a partial half-way between two grid bins must still
    come out with its true phase at the join (a causal frame skews it by ~90 degrees)."""
    L, K, _ = fit_loop_length(0.25, FS, 440.0)
    f_off = (3 * K + 0.5) * FS / L                           # between grid bins 3K and 3K+1
    n = np.arange(3 * FS)
    theta = 1.1
    x = (np.cos(2 * np.pi * 440.0 * n / FS + 0.4) + 0.5 * np.cos(2 * np.pi * f_off * n / FS + theta))[:, None]
    loop, _ = make_loop(x, FS, L, basis='dft', lock=440.0)
    J = int(1.0 * FS)
    ali = phase_align_loop(loop, x, J)
    Y = np.fft.rfft(ali[:, 0])
    m = int(np.argmax(np.abs(Y[3 * K - 2: 3 * K + 3]))) + 3 * K - 2   # where the partial was moved
    want = (theta + 2 * np.pi * f_off * J / FS) % (2 * np.pi)          # its true phase at the join
    got = np.angle(Y[m]) % (2 * np.pi)
    err = abs((got - want + np.pi) % (2 * np.pi) - np.pi)
    assert err < 0.15, np.degrees(err)


# ---------------------------------------------------------------------------- level / timbre

def test_level_match_hits_the_recording_level_at_the_join():
    x = synth_note(440.0, stereo=True)
    J = int(0.5 * FS)
    loop, _ = _loop_from(x, J, 440.0)
    loop = loop * np.array([0.3, 2.5])
    matched, g = level_match(loop, x, J, FS, 440.0)
    w = int(4 * FS / 440.0)
    for c in range(2):
        r_ref = np.sqrt(np.mean(x[J:J + w, c] ** 2))
        r_lp = np.sqrt(np.mean(matched[:, c] ** 2))
        assert abs(r_lp / r_ref - 1) < 0.05, (c, r_ref, r_lp)
    assert g[0] > 1 and g[1] < 1


def test_morph_tail_moves_the_timbre_towards_the_loop_and_is_identity_at_zero():
    f0 = 440.0
    bright = synth_note(f0, amps=(1.0, .6, .5, .4, .3), noise=0.0)
    dark = synth_note(f0, amps=(1.0, .3, .1, .05, .02), noise=0.0)
    J = int(0.6 * FS)
    loop, _ = _loop_from(bright, J, f0)
    M = int(0.3 * FS)
    tail = dark[J - M:J]

    def band_db(y):
        Y = np.abs(np.fft.rfft(y[:, 0] * np.hanning(len(y)))) ** 2
        f = np.fft.rfftfreq(len(y), 1 / FS)
        return 10 * np.log10(Y[(f > 1200) & (f < 2600)].sum() / Y[(f > 300) & (f < 600)].sum())

    target = band_db(loop)
    before = band_db(tail[-int(0.05 * FS):])
    morphed, applied = morph_tail(tail, loop, FS, max_db=9.0)
    after = band_db(morphed[-int(0.05 * FS):])
    assert morphed.shape == tail.shape
    assert abs(after - target) < abs(before - target), (before, after, target)
    assert applied > 1.0
    same, a0 = morph_tail(tail, loop, FS, max_db=0.0)
    assert np.array_equal(same, tail) and a0 == 0.0


# ---------------------------------------------------------------------------- splice / metrics

def test_splice_geometry_and_loop_points():
    f0 = 440.0
    x = synth_note(f0, noise=0.0)
    s = segment_note(x, FS)
    J = s.attack_end + int(0.1 * FS)
    loop, _ = _loop_from(x, J, f0)
    loop = phase_align_loop(loop, x, J)
    loop, _ = level_match(loop, x, J, FS, f0)
    out, ls, le, X = splice(x, s.onset, J, loop, FS, xfade_s=0.01, f0=f0)
    L = len(loop)
    assert ls == J - s.onset and le == ls + L - 1 and len(out) == le + 1
    assert X >= int(FS / f0) and X <= L // 2
    assert np.allclose(out[ls:le + 1], loop)
    n0 = int(0.003 * FS)
    assert np.allclose(out[n0:ls - X], x[s.onset + n0:J - X])
    # the loop wrap (loop_end -> loop_start) is no bigger a step than the signal's own steps
    steps = np.abs(np.diff(out[ls:le + 1, 0]))
    wrap = abs(out[ls, 0] - out[le, 0])
    assert wrap <= np.percentile(steps, 99.5)


def test_junction_prefers_the_aligned_loop():
    f0 = 440.0
    x = synth_note(f0, detune=(1.0, 1.0, 1.0023, 1.0, 1.0), noise=0.001)
    s = segment_note(x, FS)
    J = s.attack_end + int(0.1 * FS)
    loop, _ = _loop_from(x, J, f0)
    good = phase_align_loop(loop, x, J)
    bad = np.roll(good, int(FS / f0 / 2), axis=0)          # half a period out
    res = {}
    for name, lp in (('good', good), ('bad', bad)):
        lp, _ = level_match(lp, x, J, FS, f0)
        out, ls, le, X = splice(x, s.onset, J, lp, FS, xfade_s=0.01, f0=f0)
        res[name] = junction_metrics(out, x, s.onset, ls, X, FS)
    assert res['good']['score'] < res['bad']['score'], res
    assert res['good']['dip_db'] > -1.0, res['good']
    assert res['bad']['dip_db'] < res['good']['dip_db'] - 1.0, res      # a real cancellation dip
    assert res['bad']['transient_db'] > res['good']['transient_db'] + 6.0, res


def test_join_picks_phase_for_dft_and_only_rotates_for_dct():
    f0 = 440.0
    x = synth_note(f0, detune=(1.0, 1.0, 1.0023, 1.0, 1.0), stereo=True)
    s = segment_note(x, FS)
    J = s.attack_end + int(0.1 * FS)
    for basis in ('dft', 'dct'):
        loop, _ = _loop_from(x, J, f0, basis=basis)
        out, info = join(x, FS, s.onset, J, loop, f0, basis=basis, align='best')
        assert info['loop_end'] - info['loop_start'] + 1 == len(loop)
        assert len(info['gain']) == 2
        if basis == 'dft':
            assert 'phase' in info['candidates'] and info['alignment'] == 'phase', info['candidates']
        else:
            assert 'phase' not in info['candidates'] and info['alignment'] == 'rotate'
        assert info['junction']['score'] < 1.0, info['junction']


# ---------------------------------------------------------------------------- sfz

def test_key_and_tune_and_ranges():
    k, t = key_and_tune(440.0)
    assert k == 69 and abs(t) < 1e-9
    k, t = key_and_tune(467.85)
    assert k == 70 and 5 < t < 7
    ranges = assign_key_ranges([55, 70, 73])
    assert ranges == [(48, 62), (63, 71), (72, 80)]
    ranges = assign_key_ranges([70])
    assert ranges == [(63, 77)]


# ---------------------------------------------------------------------------- end to end

def test_replicate_file_end_to_end():
    f0 = 466.16
    x = synth_note(f0, stereo=True, detune=(1.0, 1.0, 1.0015, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'synth-a#4.wav')
        sf.write(src, x, FS, subtype='PCM_24')
        r = replicate_file(src, os.path.join(d, 'out'), 0.25, basis='dft')
        assert r.pitch_method == 'hint' and abs(r.f0_used - f0) < 0.1
        assert r.keycenter == 70 and abs(r.tune_cents) < 3
        assert r.alignment == 'phase'
        assert r.junction['score'] < 1.0, r.junction
        assert r.loop_metrics['max_harmonic_am_db'] < 0.5
        for k in ('audio', 'sfz', 'preview', 'json'):
            assert os.path.exists(r.outputs[k]), k
        y, fs = sf.read(r.outputs['audio'], always_2d=True)
        assert fs == FS and len(y) == r.loop_end + 1
        assert r.loop_end - r.loop_start + 1 == r.L
        txt = open(r.outputs['sfz']).read()
        assert f'loop_start={r.loop_start}' in txt and f'loop_end={r.loop_end}' in txt
        assert 'pitch_keycenter=70' in txt
        j = json.load(open(r.outputs['json']))
        assert j['loop_start'] == r.loop_start and len(j['candidates']) >= 1
        if r.sfizz is not None:
            assert r.sfizz['ok'], r.sfizz
            assert os.path.exists(r.outputs['sfizz_render'])
