# sfzc — automatic SFZ sample looping by harmonic-plus-noise resynthesis

`sfzc` turns a recorded instrument note into a compact, seamlessly looping SFZ
instrument that plays back in any standard SFZ player (sfizz, Sforzando/ARIA,
…) using only stock opcodes. It implements the two algorithms from the design
note that accompanies this directory:

* **Algorithm A** (`sfzc/looper.py`, `sfzc/harmonic.py`, `sfzc/envelope.py`):
  length-constrained resynthesis. The sustain of the note is decomposed into
  per-channel sinusoidal partial tracks plus a stochastic residual; a loop
  segment is chosen in the *parameter domain*; each partial is closed
  exactly (parameter-domain crossfade + a per-partial frequency nudge in the
  spirit of Laroche, US 6,084,170 / 6,239,345); the residual is re-synthesised
  as exactly periodic noise; the amplitude evolution is factorised into
  {constant, exp(−t/τ)} components that map one-to-one onto SFZ `ampeg_*`
  regions; brightness decay can be expressed with `fileg_*`/`cutoff`.
* **Algorithm B** (`sfzc/metric.py`): a five-term recreation-quality metric
  (spectral envelope, BS.1770 loudness envelope, seam detectability, temporal
  micro-variation, pitch/vibrato) computed on an sfizz render of the produced
  SFZ against the original note. It is also the fitness function that selects
  among loop candidates inside Algorithm A.

Everything is plain Python/NumPy/SciPy + librosa (pYIN); verification renders
use the real sfizz engine through the `pysfizz` binding.

## Backends

`--method` selects how the loop itself is built; everything around it (segmentation,
attack handling, envelopes, release, crossfade baseline, sfizz render, Metric B,
demo and report) is shared.

* **`dctloop` (default)** — the `dctloop` package in this directory: the sustain is
  rebuilt on a loop-periodic grid (cosines whose wavelengths divide the loop) with
  harmonic locking, so the loop is periodic by construction. `sfzc/dctloop_backend.py`
  runs `dctloop.loop_signal` on the note's body with the `dft` basis by default
  (`--basis dct|auto`), lines the loop up with the original at the join - either by giving
  the loop the original's phase at every grid bin measured at the join (so inharmonic
  partials are coherent too) or by rotating it, whichever gives the smaller cancellation
  dip in the cross-fade and the smaller transient at the join - level-matches it there (per-channel RMS over a 20–50 ms window at the join, so a decaying note does not step down when the loop takes over), and writes one file: recorded attack → 10 ms cross-fade → loop. Envelopes are non-negative constant/exponential regions of that file (up to three exponentials for decaying notes, so both the fast initial decay and the long tail are followed; a swell after the join is not reproduced, the loop holds the join level).
  **Tail morph** (`--no-tail-morph` to disable): the loop itself is never altered in
  spectrum; instead the last ~200 ms of the recording before the join are EQ-morphed
  towards the loop's long-term spectrum with a broad (≥ 17 bins and 1/3 octave), level-neutral
  per-band gain (≤ ±9 dB) that ramps from 0 at the start of the tail to full at the join, so
  the cross-fade blends two signals that already share a timbre. Measured as the timbre step
  across the join in excess of the recording's own step: bassoon 4.1 → 0.8 dB, clarinet
  3.8 → 1.0 dB, trumpet 2.7 → 0.8 dB.
  Decaying notes are handled specially: dctloop analyses only 2–3 loop lengths right after
  the join, level-detrended, so the loop carries the join's timbre rather than the average
  of the whole decay; a `fileg` low-pass envelope is fitted when the original darkens
  beyond that; the cross-fade is 30 ms; and under a budget several join points (attack
  end, 25 % and 40 % of the budget) are built for every class and Metric B picks one, which
  usually spends more of the budget on the recorded early decay and less on the loop. `--loop-seconds` sets the target loop length (default `0.25 + 1.25 q`,
  capped by `--max-duration`); `--lock` is dctloop's harmonic-lock width. See
  `dctloop/README.md` for the method.
* `hybrid` — the tracked-partials / original-audio hybrid path described below.
* `laroche` — the frozen loop-locked oscillator bank (`sfzc/laroche.py`).
* `auto` — hybrid vs laroche, keeping the better by Metric B.

## Usage

```bash
# loop one sample (or a directory) at quality 0.7, verify with sfizz + Metric B,
# and also build a classic crossfade loop of the same region for comparison
python3 -m sfzc loop "../Samples/Flute/flute-a4.wav" -o out -q 0.7 --baseline -v

# score any SFZ instrument against an original recording
python3 -m sfzc score original.wav instrument.sfz --key 69 --loop-len 0.5 --loop-start 0.3

# render one note with sfizz
python3 -m sfzc render instrument.sfz --key 69 --note-on 3 --dur 4 -o note.wav

# the SSO demo: recreations of ~19 instruments, sfizz renders, A/B WAVs, HTML report
python3 demo_sso.py -o demo_out -q 0.7 --play
```

Output for `name.wav`: `name.wav` (attack + loop), and for the fully
parametric path optionally `name_s2.wav` (second coherent envelope stage) and
`name_noise.wav` (residual loop); `name.sfz`; `name.json` (analysis, loop
points, metric terms). Losing
candidates are deleted; `_xfade.*` is the baseline when `--baseline` is given.

## How Algorithm A works

1. **Segmentation** (`dsp.segment_note`): onset, end of the onset transient
   (level within 15 dB of peak and |slope| < 60 dB/s, so swelling notes work),
   release onset (or, for decaying notes, the point 50 dB below peak), and a
   class: `decay` (attack < 120 ms and a fall of ≥ 6 dB within the two seconds after it, or any note falling faster than 12 dB/s by ≥ 15 dB), `oneshot` (body shorter than
   0.3 s / 16 periods, or a decaying hit shorter than 1 s or faster than
   −20 dB/s, or unpitched → written with `loop_mode=one_shot`/`no_loop`),
   else `sustain`.
2. **Pitch**: pYIN on a 22.05 kHz downmix (`harmonic.track_f0`) with an
   octave-error check on the averaged body spectrum. A note hint
   (`--note a4`) narrows the search.
3. **Harmonic analysis** (`harmonic.analyze`): zero-phase Blackman–Harris
   STFT, ~10 periods long, hop ≤ 5.8 ms. For every **channel separately**
   (the SSO "stereo" samples carry differently detuned content left/right)
   and every harmonic k the peak nearest `k·f0·√(1+Bk²)` is refined by
   parabolic interpolation → amplitude, frequency and phase tracks. f0 is
   re-estimated per frame from the strongest partials; the inharmonicity
   coefficient B is estimated per note and the analysis re-run with it.
   Tracks are low-passed (zero phase, 25/40 Hz) so estimation jitter does
   not become AM/FM sidebands at synthesis.
   A harmonic model needs a dense low harmonic series (≥ 5 of the first 12 and
   ≥ 3 of the first 6 within 35 dB of the strongest partial, centroid < 30·f0);
   sparse spectra (vibraphone, chimes, crotales, celeste …) go to
   `harmonic.analyze_peaks`, an **inharmonic peak-tracking** model: the
   strongest stable peaks of the body spectrum are tracked individually and
   the rest of the pipeline is unchanged (loop lengths are multiples of the
   lowest partial's period; every partial is phase-closed separately).
4. **Residual** (`harmonic.residual_analysis`): a 4× longer window resolves
   the between-harmonic floor; harmonic main lobes of all channels are masked
   and interpolated across; kept as mid/side magnitude spectrograms.
5. **Loop search** (`looper._seam_cost_search`): over (start, length) on the
   frame grid, minimising the loudness-weighted mismatch of the spectral
   shape and partial frequencies between the context after the loop start
   and after the loop end, plus the size (in cents) of the frequency nudge
   each partial would need to close its phase over that length. Lengths are
   bounded by the quality knob.
6. **Loop closing** (`looper.build`): amplitude tracks are detrended over the
   loop (the trend goes to the envelope); the last 25 % of each track is
   cross-faded — in the parameter domain — towards the material that precedes
   the loop start (or towards the start value), and the cumulative phase of
   every partial is closed with a linear phase term (a frequency shift
   ≤ 0.5/L Hz, reported in cents). The loop then tiles *exactly*, for every
   channel, without any waveform crossfade. The synthesis is anchored to the
   analysed phase at the loop start so the original attack cross-fades into it
   without comb filtering.
7. **Envelope factorisation** (`envelope.fit_envelope_components`): partial
   amplitudes over the body are fitted as `Σ_j a_j[k]·e_j(t)` with
   `e ∈ {1, exp(−t/τ)}` (NNLS for decaying notes, a grid over τ). Each
   component becomes its own region playing a phase-identical loop with that
   partial-amplitude vector, so the regions sum coherently and the timbre
   morphs over time (piano: bright fast component + dark slow component; up to
   three stages at q ≥ 0.75, τ's found by coarse grid + coordinate descent,
   errors weighted by each partial's audibility per frame). sfizz
   and ARIA both implement `ampeg_decay` as `exp(−9t/T)` when
   `ampeg_sustain=0`, so `T = 9τ` and no `*_shape` opcode is needed. Held
   sustains keep at least 15 % of the loop-start level. The residual gets its
   own region(s) whose envelope follows the recorded residual level.
8. **Hybrid mode** (`looper.build_hybrid`, default for sustaining notes with
   loops ≥ 6 periods, `--hybrid off` to disable): the first ~70 % of the loop
   is the *original* audio (broadband-detrended), and only a bridge at the end
   is resynthesised: it starts phase-locked to the analysed partials, is
   cross-faded in over a few ms (no comb filtering, since it is phase
   aligned), and its partials are phase-closed onto the analysed phases at
   the loop start, so the wrap is exact. The broadband level evolution is
   fitted as constant + exponential and both are regions of the *same* file
   (coherent by construction). This keeps the modelling error out of most of
   the loop and needs a single WAV. Decaying notes keep the fully
   parametric path because their per-partial detrending and coherent
   timbre-morphing stages need synthesised loops.
9. **Verification**: the SFZ is rendered with sfizz (`render.render_sfz`,
   unity-gain corrected) for the original note's duration and scored with
   Metric B; the best of `--candidates` loops is kept.

### The quality knob q ∈ [0, 1]

| q | loop-length budget (all files together) | partials | residual | envelope stages |
|---|---|---|---|---|
| 0 | 2 periods (≥ 5 ms) | 12 | off | 1 |
| 0.5 | ~35 % of the usable body | ~12 + 43 % of the rest | on | 2 |
| 0.75+ | ~65 % of the usable body | most | on | 3 (decaying notes) |
| 1 | 80 % of the usable body | all | on | 3 |

The budget is in samples on disk, so the parametric path (one file per
envelope stage + one for the residual) gets proportionally shorter loops than
the single-file hybrid path. Loop lengths are whole numbers of fundamental
periods; the search returns the best candidate of every factor-2 length band
and Metric B arbitrates (`--candidates`, default 4) - a small rate-distortion
search under the q budget.

### Hard duration budget

`--max-duration 0.5` (`LoopConfig.max_total_s`) caps the **total audio written
for a sample, all files together**, at 0.5 s. Under a budget:

* the loop may start right after the onset transient (≤ 100 ms after the
  onset) instead of waiting for the level to settle;
* every candidate must satisfy `(loop start − onset) + n_files · N ≤ budget`;
* several configurations are built and scored by Metric B: for sustaining
  notes the single-file hybrid path and the parametric path with 1–2 stages,
  for decaying notes 1–3 stages with/without the residual file;
* candidates are drawn from different loop-length bands (longest first) so
  the metric can arbitrate between a longer loop and more stages;
* one-shots that do not fit are looped if pitched (harp, bass drum), or
  truncated with a 20 ms fade if unpitched.

Extra stage and noise files hold only the loop (plus a few ms of pre-roll)
and are started with the `delay` opcode, faded in with `ampeg_attack`. sfizz
starts a delayed voice exactly 2 samples late (measured, independent of block
size), which is compensated so all stages still sum sample-coherently;
`--stage-files padded` writes zero-filled attacks instead (player agnostic,
bigger). The 19-sample demo under a 0.5 s budget lands at 12–87 kB per sample
(originals 282–1520 kB) in about 70 s total.

## The loop-locked (Laroche) method — `sfzc/laroche.py`

A second implementation of Algorithm A, following the "Reconstructing sustain
segments into exactly periodic loops" report, selectable with
`--method laroche` (`--method auto`, the default, runs both and keeps the one
Metric B prefers):

* **Frozen, loop-locked partials.** Every partial is frozen to its loop-region
  statistics (median frequency per channel, detrended mean amplitude, analysed
  phase at the loop start) and locked to the loop's DFT grid: `k = round(f·L/fs)`,
  `f' = k·fs/L` — for a stationary partial this is exactly Laroche's minimal
  phase-closing frequency nudge. The oscillator-bank resynthesis is therefore
  periodic by construction (integer cycles, verified to 1e-12 by `sfzc diagnose`).
* **L selection** by an energy- and JND-weighted detuning cost
  (JND ≈ 3 Hz below 500 Hz, 0.6 % above), a mild pull towards Laroche's
  ~166 periods, a vibrato-cycle term, and the size budget; candidates are drawn
  from every factor-2 length band and arbitrated by Metric B.
* **Loop start** chosen among a few positions after the attack by the quality
  of the coherent envelope-stage fit, so struck notes start after their
  non-exponential prompt decay.
* **Residual as filtered noise in its own loop** whose length differs from the
  harmonic loop, so the two repetition periods never coincide (combined period
  = lcm). Under a budget the noise loop takes whatever samples remain.
* **DC removal**, seam diagnostics (tiled spectral flux at the seam vs the
  interior, sample continuity, DC, integer-cycle and detuning checks) stored in
  the JSON and shown on the report page; `python3 -m sfzc diagnose x.sfz` runs
  them on any SFZ loop.
* **Stage 2 opcodes**: `--lfo` (gentle `pitchlfo`/`amplfo`), `--loop-crossfade S`
  (sfizz/OpenMPT safety net), `--round-robin 2` (a second loop set from later
  in the sustain via `seq_length`/`seq_position`).
* **Stage 3** `--refine`: PyTorch optimisation of per-partial and per-noise-band
  gains against a multi-resolution, level-normalised log-STFT loss between the
  tiled loop and the source around the loop start (frequencies stay locked).
  On the SSO material it lowers its own loss but not Metric B, so it is off by
  default.
* Sustains with vibrato/tremolo (> 3 cents / > 1.5 dB) are handed to the
  tracked parametric path (parameter-domain closing keeps the movement), as the
  report recommends; `--frozen on` forces freezing.

Measured on the demo set the two methods are complementary: the loop-locked
path wins on decaying/mallet notes and under tight budgets (piano at 0.5 s:
0.40 vs 0.15), the tracked/hybrid path on long sustains with room (horn 0.58 vs
0.48). One caveat specific to this library: its "stereo" samples carry
detuned left/right content that beats; a stationary model flattens that beat
(the tracked path reproduces it as micro-variation), and with loops shorter
than ~1 s the grid cannot keep both channels' fundamentals on it.

## Algorithm B in detail

`metric.evaluate_recreation(original, rendered, sr, held_range, loop_period_s, loop_start_s)`

* `D_spec` — mean |dB| difference of 96-band log-mel spectra (1024/4096 FFTs),
  both per frame and averaged over 250 ms blocks, plus a linear-frequency
  log-STFT term; frames weighted by level.
* `D_loud` — mean |LU| difference of BS.1770 momentary loudness contours after
  global level matching, plus 0.25·|gain offset|.
* `D_seam` — (a) novelty (half-wave rectified log-mel flux, high-passed at
  ~20 Hz so only transients count) at the known seam phase relative to the
  90th percentile of all loop phases, in dB, plus the one-time spectral step at the
  attack→loop junction (`junction_db`: mean |dB| log-mel difference between the 40 ms
  before and after the join, in excess of the original's own step there); (b) periodic level *pumping*:
  std of the loop-cycle-folded log envelope minus the same for the original;
  (c) a small weight on the excess autocorrelation of the transient novelty
  at the loop lag.
* `D_var` — under-variation (and, with lower weight, over-variation) of
  detrended MFCC/centroid/RMS trajectories in the held region: a "dead"
  loop scores badly.
* `D_pitch` — median pitch offset, vibrato depth and vibrato rate mismatch
  (pYIN, 2-cent resolution).

`score = exp(−Σ w_i D_i)` with default weights `spec 1/9, loud 1/6, seam 2.5,
var 0.6, pitch 0.12`. **These defaults are uncalibrated engineering values**;
`metric.calibrate(rows, ratings)` fits them by NNLS to listening-test ratings.
In practice ~0.45–0.6 is a transparent recreation of a natural, evolving
note (the original itself, looped by crossfade, lands in the same range),
0.2–0.4 is audible-but-decent, < 0.15 is clearly wrong.

## sfizz facts established empirically (probe scripts, see git history)

* `ampeg_decay`/`ampeg_release`: pure exponential, −78.2 dB per T seconds
  (`exp(−9t/T)`), sustain is a floor; `ampeg_attack` is linear in amplitude;
  `fileg_*` uses the same exponential in cents; `ampeg_*_shape` is ignored;
  `loop_end` is inclusive; `loop_crossfade` works; the engine applies a fixed
  −11.5 dB headroom gain, which the renderer wrapper compensates.

## Speed

Measured on this laptop (single process, default settings: 4 verified
candidates + crossfade baseline + A/B renders): 1–5 s for most samples, 8 s for
a 14-second vibraphone note; the whole 19-sample demo runs in about a minute.
With `--jobs 6` the 10-file Oboe directory takes 3.3 s wall-clock (about 1.8 s
of CPU per sample). `--no-verify` (take the search's best loop without
rendering) is 2–3× faster again.

What made it fast: pitch by harmonic summation on the body spectrum instead of
pYIN (`--f0 pyin` restores it, ~3 s per call); the metric's vibrato terms use
the instantaneous frequency of the strongest low partial (Hilbert) instead of
a pitch tracker on every render; the partial picker and residual masking are
vectorised per frame; partials that never come within 60 dB of the strongest
are pruned before synthesis; track interpolation works per frame segment; mel
filterbanks are cached and the envelope fit solves all partials as one batched
least-squares problem.

## Tests

`python3 tests/test_synthetic.py` (or pytest): analysis accuracy on a
synthetic stereo tone, envelope factorisation recovery, exact-periodicity +
seam + score checks on a synthetic vibrato note, and the one-shot path.

## Known limitations / next steps

* Polyphonic or chordal samples and unpitched material are not modelled
  (they fall back to one-shots).
* The release tail is expressed with `ampeg_release` only; re-synthesising the
  recorded tail behind `loop_sustain` is not implemented.
* Metric weights are uncalibrated (no listening test yet); `calibrate()` is
  the hook for that.
* Very low notes (< ~50 Hz) get windows shorter than the ideal 10 periods.
