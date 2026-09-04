"""dctjoin.bridge: the attack's harmonics land exactly on the untouched loop's at the join."""
import numpy as np
import pytest

from dctloop import loop_signal
from dctjoin.bridge import bridge_attack, continuity_metrics, find_join_bridge, harmonic_tracks, loop_harmonics
from dctjoin.join import splice

FS = 44100


def living_note(f0=440.0, secs=3.5, amps=(1.0, .5, .3, .2, .12, .08), bright=(0.0, 0.5, 1.5, 2.0, 2.5, 3.0),
                vib_cents=15.0, vib_hz=5.5, trem_db=2.0, trem_hz=4.0, detune=(1.0, 1.0), noise=0.003, seed=0):
    """A note whose early harmonics are brighter than the body (decaying with tau 0.4 s), with vibrato
    and tremolo throughout and an optional per-channel detune.  Level 0.2 so that even the bright
    start (up to 4x the body's harmonic amplitudes) stays below full scale."""
    rng = np.random.default_rng(seed)
    n = int(secs * FS)
    t = np.arange(n) / FS
    env = np.minimum(1.0, (t / 0.06) ** 2)                            # 60 ms attack ramp
    trem = 10 ** (trem_db / 20 * np.sin(2 * np.pi * trem_hz * t))
    chans = []
    for c, mult in enumerate(detune):
        inst_f = f0 * mult * (1 + (vib_cents / 1200) * np.log(2) * np.sin(2 * np.pi * vib_hz * t + c))
        ph = 2 * np.pi * np.concatenate([[0.0], np.cumsum(inst_f)[:-1]]) / FS      # phase at n = 2 pi sum_{i<n} f_i / fs
        y = np.zeros(n)
        for h, (a, b) in enumerate(zip(amps, bright), 1):
            a_t = a * (1 + b * np.exp(-t / 0.4))
            y += a_t * np.cos(h * ph + 0.7 * h + c)
        chans.append(0.2 * env * trem * y + noise * rng.standard_normal(n))
    return np.stack(chans, 1)


def test_harmonic_tracks_recover_amplitude_frequency_and_phase():
    x = living_note(vib_cents=0.0, trem_db=0.0, bright=(0,) * 6, noise=0.0)
    centres = np.arange(int(1.0 * FS), int(1.2 * FS), 441)
    tr = harmonic_tracks(x, FS, 440.0, 6, centres)
    want = 0.2 * np.array([1.0, .5, .3, .2, .12, .08])
    assert np.allclose(tr['amp'][:, :, 0].mean(axis=0), want, rtol=0.03), tr['amp'].mean(axis=0)
    assert np.allclose(tr['freq'][:, :, 0].mean(axis=0), 440.0 * np.arange(1, 7), atol=0.5)
    # total phase at a frame centre equals the synthetic phase there (mod 2 pi)
    k = 3
    got = tr['phase'][k, 0, 0] % (2 * np.pi)
    exp = (2 * np.pi * 440.0 * centres[k] / FS + 0.7) % (2 * np.pi)
    assert abs((got - exp + np.pi) % (2 * np.pi) - np.pi) < 0.05


def test_loop_harmonics_read_the_loops_grid():
    x = living_note(vib_cents=0.0, trem_db=0.0, bright=(0,) * 6, noise=0.0)
    loop, info = loop_signal(x[int(1.0 * FS):], FS, 0.5, basis='dft', f0=440.0)
    lh = loop_harmonics(loop, FS, 440.0, 6)
    assert np.allclose(lh['f'][:, 0], np.arange(1, 7) * info['K'] * FS / info['L'])
    assert np.allclose(lh['A'][:, 0], 0.2 * np.array([1.0, .5, .3, .2, .12, .08]), rtol=0.05)


@pytest.mark.parametrize('detune', [(1.0, 1.0), (1.0, 1.0035)])
def test_bridge_makes_the_join_continuous_where_a_splice_is_not(detune):
    x = living_note(detune=detune)
    f0 = [440.0 * d for d in detune]
    loop, info = loop_signal(x[int(1.2 * FS):], FS, 0.5, basis='dft', f0=f0)
    J = int(0.5 * FS)
    # plain splice at the same J (no bridge): the early harmonics are brighter, the phases are wherever
    out0, ls0, _, X0 = splice(x, 0, J, loop, FS, xfade_s=0.01, f0=440.0)
    c0 = continuity_metrics(out0, x, J, ls0, FS, f0)
    # bridged
    xb, binfo = bridge_attack(x, FS, J, loop, f0, bridge_s=0.3)
    out1, ls1, le1, X1 = splice(xb, 0, J, loop, FS, xfade_s=0.01, f0=440.0)
    c1 = continuity_metrics(out1, x, J, ls1, FS, f0, loop=loop)
    # nothing before the bridge changed, the loop is untouched
    assert np.array_equal(xb[: binfo['bridge_start']], x[: binfo['bridge_start']])
    assert np.array_equal(out1[ls1: le1 + 1], loop)
    # the splice alone steps; the bridge does not.  Sample domain: the bridged tail IS the loop's entry
    assert c0['harm_step_db_wmean'] > 1.5 or c0['harm_phase_err_deg_wmean'] > 20, c0
    assert c1['tail_ncc'] > 0.995, c1
    assert c1['tail_residual_db'] < -25, c1
    # and per harmonic no worse than the loop against itself (vibrato makes bands wander by nature)
    assert c1['excess_harm_step_db'] < 0.3, c1
    assert c1['excess_phase_err_deg'] < 5.0, c1
    assert binfo['amp_move_db_weighted'] > 1.0          # it really had something to do
    assert binfo['model_fit_db'] < -25                   # the harmonic model describes the tail well


def test_bridge_adds_no_click_at_either_end():
    x = living_note()
    loop, _ = loop_signal(x[int(1.2 * FS):], FS, 0.5, basis='dft', f0=440.0)
    J = int(0.5 * FS)
    xb, binfo = bridge_attack(x, FS, J, loop, 440.0, bridge_s=0.3)
    out, ls, _, _ = splice(xb, 0, J, loop, FS, xfade_s=0.01, f0=440.0)
    d2 = np.diff(out.mean(1), 2) ** 2
    w = int(0.003 * FS)
    ref = np.median([d2[i: i + w].mean() for i in range(ls + w, len(d2) - w, w)])
    for p in (binfo['bridge_start'], ls):
        assert d2[p - w // 2: p + w // 2].mean() < 3 * ref, p


def test_find_join_bridge_prefers_where_the_timbre_already_matches():
    x = living_note()
    loop, _ = loop_signal(x[int(1.2 * FS):], FS, 0.5, basis='dft', f0=440.0)
    lo, hi = int(0.4 * FS), int(1.6 * FS)
    fj0 = find_join_bridge(x, FS, loop, 440.0, lo, hi, time_weight_db_per_s=0.0)
    assert fj0['J'] > int(0.9 * FS), fj0                # brightness has decayed by then
    fj1 = find_join_bridge(x, FS, loop, 440.0, lo, hi, time_weight_db_per_s=20.0)
    assert fj1['J'] < fj0['J']                          # a strong time preference pulls it earlier
    assert fj0['amp_distance_db'] <= fj1['amp_distance_db']


def test_attack_budget_is_honoured_and_the_join_stays_continuous():
    """living_note's brightness decays with tau 0.4 s, so the free search joins late; with a
    0.5 s budget the join must land inside it and still be exact."""
    import os, tempfile
    import soundfile as sf
    from dctjoin.unaltered import replicate_unaltered
    x = living_note(secs=4.0)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'viola-a4.wav')
        sf.write(src, x, FS, subtype='PCM_24')
        free = replicate_unaltered(src, None, 0.5, max_attack_s=None, search_s=1.5, sfizz=False, preview=False)
        tight = replicate_unaltered(src, None, 0.5, max_attack_s=0.5, search_s=1.5, sfizz=False, preview=False)
        assert free.attack_s > 0.5
        assert tight.attack_s <= 0.5 + 1e-6, tight.attack_s
        assert tight.bridge['amp_move_db_weighted'] >= free.bridge['amp_move_db_weighted']   # more to morph, earlier
        for r in (free, tight):
            assert r.loop_untouched
            assert r.continuity['tail_ncc'] > 0.99, r.continuity
            assert r.continuity['excess_harm_step_db'] < 0.3 and r.continuity['excess_phase_err_deg'] < 5.0, r.continuity
        # a budget tighter than the transient + minimum bridge still works (bridge overlaps the transient tail)
        tiny = replicate_unaltered(src, None, 0.5, max_attack_s=0.15, search_s=1.5, sfizz=False, preview=False)
        assert tiny.attack_s <= 0.15 + 1e-6 and tiny.loop_untouched and tiny.continuity['tail_ncc'] > 0.98
