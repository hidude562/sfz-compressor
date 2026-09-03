# dctloop — seamless loops of stationary sounds

Give `dctloop` any number of seconds of a sustained, (assumed) constant sound and it
returns a loop that closes **by construction**: the loop is rebuilt from cosines whose
wavelengths all divide the loop length, so there is no seam to crossfade. Two synthesis
bases are offered, `dct` and `dft`; they share one analysis.

Plain Python: NumPy, SciPy, soundfile, librosa (pYIN only). Tested on the Sonatina
Symphonic Orchestra trumpets, solo violin and cello, and violin sections.

## Use

```python
from dctloop import loop_signal, loop_file

# an array of steady sustain (N,) or (N, C); returns the loop and an info dict
loop, info = loop_signal(x, fs, seconds=1.5, basis='dct')
loop, info = loop_signal(x, fs, seconds=0.25, basis='dft', f0=466.2)   # pitch known

# a recorded note on disk: sustain found automatically, files written to out/
r = loop_file('Samples/trumpet/trumpet-a#4.wav', 'out', seconds=0.25, basis='dft')
r.outputs['loop']                     # out/trumpet-a#4_loop.wav (24-bit)
r.metrics['seam_flux_ratio']          # ~1: the seam is invisible
```

Command line (same options):

```bash
python3 -m dctloop "Samples/trumpet/trumpet-a#4.wav" -o out --loop 0.25 --basis dft --play
python3 -m dctloop "Samples/1st violins" -o out --loop 1.5       # a whole directory
python3 dctloop/run_demo.py --loop 0.25 --basis dct,dft --play    # the SSO test set + summary.md
python3 -m pytest dctloop/tests -q
```

Per input, `loop_file` writes `<name>_loop.wav` (the bare loop), `<name>_preview.wav`
(2 s of the original sustain, a gap, then the loop repeated for 4 s) and `<name>.json`
(everything in the `LoopResult`).

### Options

| option | default | meaning |
|---|---|---|
| `seconds` / `--loop` | 1.5 | target loop length; rounded to an integer number of periods, shortened if the sustain is shorter than two loops |
| `periods` / `--periods` | – | instead of `seconds`: exactly this many fundamental periods |
| `basis` / `--basis` | `dct` | `dct`: real cosines, palindromic loop. `dft`: the input's phases, no mirror |
| `f0` / `--f0` | auto | fundamental in Hz; auto = note name in the file name as a hint, pYIN, then spectral refinement to ~0.01 cent |
| `lock` / `--lock` | 1.5 | harmonic-lock half-width in grid bins (0 disables) |
| `phase` / `--phase` | `orig` | partial phases (dct: signs) from the input, or `random` |
| `fit` / `--no-fit` | on | round the loop to whole periods |
| `start`, `dur` / `--start --dur` | auto | analysis segment in seconds instead of automatic sustain detection |

Choosing a loop length: **0.25 s** gives a static, perfectly stable tone (vibrato and
chorus are frozen out, 14–17 analysis frames average the noise floor cleanly). **1.5 s**
keeps a slow shimmer from the original's motion at the cost of only 1–3 analysis frames.
Anything in between trades one for the other.

## How it works

1. **Sustain** (`find_body`): the longest stretch where the smoothed RMS envelope stays
   within 12 dB of its peak, trimmed 150 ms at each end.
2. **Pitch** (`pitch.py`): pYIN per channel (SSO files are stereo and some sections carry
   different detunings left and right), then refined by parabolic interpolation of the
   first six harmonic peaks. pYIN alone is quantised to 10 cents; the loop fit needs
   ~0.01 cent.
3. **Loop length** (`fit_loop_length`): an even number of samples `L` near the target that
   holds an integer number `K` of fundamental periods as exactly as possible. Harmonic *h*
   then sits on grid bin `h·K`. The grid spacing is `fs/L` Hz (0.67 Hz at 1.5 s, 4 Hz at
   0.25 s). The analysed segment must be at least `2L` long.
4. **Analysis on the loop grid** (`analyse_on_grid`). A real cosine transform cannot measure
   a partial's amplitude on its own — one DCT bin holds only the cosine projection, the sine
   part leaks into the neighbours — so the analysis uses the DCT with its quadrature twin,
   the DST (together, the DFT), and only the synthesis is a pure cosine transform. It is a
   Welch periodogram on frames of exactly `2L` samples with a periodic Hann window, whose
   spectrum is zero at every multiple of `1/L`: an on-grid partial lands in exactly one grid
   bin whatever its phase. The averaged periodogram is then distributed over the grid:
   * **harmonic locking** — everything within ±`lock` grid bins of `h·f0`, plus the falling
     skirt beyond, goes to bin `round(h·f0·L/fs)`;
   * every other spectral peak is moved **whole** to its nearest grid bin (never split
     between two bins, which would beat at `1/L` Hz);
   * the remaining noise floor is shared between neighbouring bins in proportion to their
     energy.

   Every bin is used once, so the loop has the input's power spectrum on the grid. The
   calibration is exact: white noise comes back at unity gain, a harmonic of amplitude 1 as
   1.000. Noise is kept, "frozen" into a texture with period `L`.
5. **Synthesis** (`synthesise`).
   * `dct`: amplitudes with a sign into an inverse DCT-II of `L/2` points. Basis function *m*
     is `cos(π·m·(2n+1)/L)`, wavelength `L/m`, i.e. *m* whole cycles per loop. Every such
     cosine is even about `n = −½` and `n = L/2 − ½`, so the loop is a **palindrome**: the
     second half mirrors the first. A cosine can only carry a sign, so the *relative* phase
     between channels is rounded to 0 or π against channel 0, keeping whichever gives the
     mono sum closest to the original's (a partial only goes anti-phase when it was more
     than ~120° apart). Choosing each channel's sign independently scrambles the stereo
     image and cancels partials in mono.
   * `dft`: the same amplitudes with the input's complex phases, inverse real FFT, no mirror.
     The original waveform shape is kept.

   Finally the loop is scaled to the RMS of the analysed segment; the reported `gain` is
   ≈1 when the calibration holds.

### Why harmonic locking

Any energy on the grid bins next to a harmonic (bin `h·K ± 1`) beats with that harmonic at
exactly `1/L` Hz, and the DCT phase-locks every partial to the seam, so the beat becomes a
clean amplitude swell once per loop ("wah"). Measured on the SSO trumpet at 0.25 s the
neighbours of harmonic 3 were only 6.6 dB below it (an 8.8 dB swell), because the
player's pitch drifts a few cents over the 3.5 s analysis and the averaged spectrum smears
each harmonic over several 4 Hz bins; a string section's chorus does the same. With the
lock, harmonics come out perfectly stable and exactly harmonic; only the real noise floor
between them keeps its `L`-periodic texture. At long loops the lock window is narrow in Hz
(±1 Hz at 1.5 s), so drift energy further out remains as a slow, natural-sounding shimmer.

## Metrics (`metrics.py`, in `LoopResult.metrics` and the demo summary)

* `seam_flux_ratio`, `mid_flux_ratio`: spectral flux at the loop seam (and the palindrome
  mid-point) over the median flux of the loop. ≈1 means the join is no more eventful than
  any other instant; `p95_flux_ratio` gives the scale of normal variation.
* `max_harmonic_am_db`: the worst swell at the loop rate over the first ten harmonics
  (0 with locking).
* `ltas_mean_abs_db` / `ltas_max_abs_db`: third-octave long-term spectrum of the loop vs the
  analysed segment, per channel; `mono_db`: level of the mono sum, sensitive to how the
  inter-channel phases were frozen.
* `info['grid_offset']`: energy-weighted distance of the strongest peaks from the grid before
  assignment (0 = all on grid, 0.25 = random); informative only.

Typical values on the SSO test set (0.25 s loops, either basis): seam× 1.0–2.0 against
p95× 1.4–2.1, spectrum error 0.1–0.5 dB mean, wah 0.0 dB.

## Limits, by design

* The loop can only contain frequencies `m·fs/L`. Anything else is moved to the nearest
  grid line (≤ `fs/2L` Hz, inaudible as pitch), and any vibrato, tremolo or chorus becomes
  a texture that repeats every `L` seconds. That is the "assumed constant" in the brief.
* A DCT loop is a palindrome: `L/2` samples of unique material, with a time reversal at
  the seam and at the middle. For stationary content this is inaudible; `dft` is there to
  A/B it.
* The analysis needs `≥ 2L` of steady sustain (`loop_signal` shortens the loop otherwise).

## Layout

| file | content |
|---|---|
| `core.py` | `loop_signal`, `make_loop`, `analyse_on_grid`, `synthesise`, `fit_loop_length` — the method |
| `pitch.py` | note names, pYIN, spectral refinement |
| `metrics.py` | seam, wah and spectrum measurements |
| `pipeline.py` | `loop_file`, sustain detection, preview and file output, `LoopResult` |
| `cli.py` | `python3 -m dctloop` |
| `split.py` | experimental: a short harmonic loop plus a long residual loop as two SFZ regions (`--split`) |
| `run_demo.py` | the SSO test set with a summary table |
| `tests/` | synthetic checks of every claim above |
