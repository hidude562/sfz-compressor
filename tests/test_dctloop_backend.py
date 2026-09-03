"""The dctloop backend wrapped into an SFZ recreation: periodic loop, coherent join, sane score."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sfzc import dsp, looper  # noqa: E402
from test_synthetic import SR, synth_note  # noqa: E402


def test_dctloop_backend_loop_and_join():
    x = synth_note(vib_cents=4.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tone-a3.wav")  # the name carries the (correct) pitch
        dsp.write_audio(p, x, SR)
        r = looper.process_sample(p, d, looper.LoopConfig(q=0.6, method="dctloop", baseline=True))
        assert r.klass == "sustain" and r.info["mode"].startswith("dctloop")
        w, _ = dsp.load_audio(r.wav_paths[0])
        ls, le = r.loop_start, r.loop_end
        loop = w[ls: le + 1]
        # exactly periodic by construction: wrap step no larger than typical steps
        assert np.abs(loop[0] - loop[-1]).max() < 1.5 * np.percentile(np.abs(np.diff(loop, axis=0)), 99)
        # the join is coherent: no transient at loop_start in the file (novelty within the loop's range)
        m = dsp.to_mono(w)
        hop = 256
        from sfzc.metric import _novelty
        nov = _novelty(m, SR, hop)
        j = ls // hop
        assert nov[max(0, j - 1): j + 2].max() < np.percentile(nov[j + 4: (le // hop)], 99) * 1.5 + 1e-6
        assert r.metric["score"] > 0.25, r.metric
        assert r.metric["score"] >= r.baseline_metric["score"] * 0.8


if __name__ == "__main__":
    test_dctloop_backend_loop_and_join()
    print("ok test_dctloop_backend_loop_and_join")
