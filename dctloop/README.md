# dctloop — seamless loops of a stationary sound by DCT reconstruction

Give it any number of seconds of a sustained, (assumed) constant sound and it
returns a loop that closes **by construction**: the loop is rebuilt from
cosines whose wavelengths all divide the loop length exactly, so there is no
seam to crossfade.

```bash
# one file, 1.5 s loop, write loop + A/B preview into dctloop_out/, and play it
python3 -m dctloop "../Samples/trumpet/trumpet-a#4.wav" -o dctloop_out --loop 1.5 --play

# the SSO test set (trumpets, solo violin, cello, two violin sections), all variants
python3 dctloop/run_demo.py -o dctloop_out --loop 1.5 [--play]

# harmonic loop (a few periods) + 3 s residual loop + two-region SFZ, verified in sfizz
python3 -m dctloop "../Samples/violin/violin-a#4.wav" -o out --split [--harm-bw 30] [--resid-loop 3] --play
python3 dctloop/run_demo.py -o dctloop_out_split --split [--harm-bw 30] [--play]

python3 -m pytest dctloop/tests -q
```

Outputs per note: `<name>_loop.wav` (the bare loop, 24-bit), `<name>_preview.wav`
(2 s of the original sustain, a gap, then the loop repeated for 4 s) and
`<name>.json` (pitch, loop length, metrics).

## How it works

1. **Sustain.** The longest stretch where the smoothed RMS envelope stays within
   12 dB of its peak, trimmed by 150 ms at each end (`find_body`), or `--start/--dur`.
2. **Pitch.** pYIN per channel (SSO files are stereo and some sections carry
   different detunings left/right), then refined to ~0.01 cent by parabolic
   interpolation of the first six harmonic peaks (`refine_f0`).
3. **Loop length.** `--loop T` is rounded to an even number of samples `L` that
   holds an integer number `K` of fundamental periods as exactly as possible
   (`fit_loop_length`). Harmonic *h* then sits on grid bin *h·K*. The loop grid is
   `fs/L` Hz (0.67 Hz for 1.5 s). The analysed segment must be at least `2L`
   long; if the sustain is shorter the loop is shortened.
4. **Analysis on the loop grid** (`analyse_on_grid`). A real cosine transform on
   its own cannot measure a partial's amplitude — DCT bin *k* holds only the cosine
   projection, the sine part leaks into the neighbouring bins — so the analysis
   uses the DCT *with* its quadrature twin, the DST (together: the DFT), and only
   the synthesis is a pure cosine transform.
   * `--mode snap` (default): Welch periodogram on frames of exactly `2L`
     samples with a periodic Hann window. That window's spectrum is zero at every
     multiple of `1/L`, so an on-grid partial lands in exactly one grid bin
     whatever its phase. Every local spectral maximum is treated as a peak and its
     whole main lobe is moved to the *nearest* grid bin (at most `fs/2L` Hz away)
     instead of being split into two components that would beat once per loop.
     The remaining bins (noise floor) are shared between neighbours in proportion
     to their energy. Every bin is used once, so the loop has the same power
     spectrum as the input on the grid: noise is kept, "frozen" into a texture
     with period `L`. The calibration is exact (unity gain on white noise; a
     harmonic of amplitude 1 comes back as 1.000).
   * `--mode comb`: one Hann frame over the whole `N = q·L` segment, sampled at
     bins `q·m` only — the frequency-domain view of folding the signal at period
     `L` and averaging. Also exact for on-grid partials, but off-grid content
     (noise, detuned partials) loses `10·log10(q/1.5)` dB. A gentle de-noiser.
5. **Synthesis.**
   * `--basis dct` (default): amplitude `a_m` (with a sign) goes into an inverse
     DCT-II of `L/2` points. Basis function *m* is `cos(π·m·(2n+1)/L)`: wavelength
     `L/m` samples, i.e. *m* whole cycles per loop. Every such cosine is even about
     `n = -½` and about `n = L/2 - ½`, so the loop is a **palindrome**: the second
     half is the mirror image of the first, and the `L`-sample loop is the `L/2`-point
     inverse transform followed by its reversal. A cosine can only carry a sign, so
     the *relative* phase between channels is rounded to 0 or π against channel 0
     (choosing each channel's sign on its own scrambles the stereo image and
     cancels partials in mono).
   * `--basis dft`: the same grid amplitudes with the input's complex phases, no
     mirror. Same seam guarantee, the waveform shape of the original is kept.
   * `--phase random` draws the signs/phases instead of taking them from the input.

   Finally the loop is scaled to the RMS of the analysed segment; the reported
   `gain` is ≈1 when the calibration is right and grows in `comb` mode by the
   noise it removed.

## What the numbers mean

* `seam×` / `mid×`: spectral flux at the loop seam (and at the palindrome's
  mid-point) divided by the median flux of the loop. ≈1 means the join is no
  more eventful than any other instant; `p95×` is the 95th percentile for scale.
* `ltas mean/max dB`: third-octave long-term spectrum of the loop against the
  analysed segment (mono sum).
* `grid-off`: energy-weighted distance of the 20 strongest peaks from the loop
  grid, per channel (0 = every partial on a grid line, 0.25 = random). Only
  informative — peaks are moved to the grid either way.
* `det`: left/right detune in cents (refined f0 per channel).

## Two loops instead of one: `--split`

A single loop of length `L` freezes *everything* that is not a steady harmonic
(noise, chorus, vibrato sidebands) into a texture that repeats every `L`
seconds; below ~1.5 s that repetition is heard as a "swish" at `1/L` Hz. The
harmonics themselves have period `1/f0` and never restart. `dctloop/split.py`
therefore produces (`process_split`):

* `<name>_harm.wav` — the grid lines at multiples of `K` (the `h·f0` partials)
  rendered as a loop of a few periods (the shortest integer number of periods
  within 0.5 cent, typically 400–1300 samples), with the original phases and
  inter-channel relations. The leftover pitch error is written as `tune=` in the
  SFZ, so the sampler plays the exact measured f0.
* `<name>_resid.wav` — everything else, re-gridded (power-density interpolation)
  onto its own loop length (`--resid-loop`, default 3 s) with random phases: a
  fresh noise realisation with the analysed spectrum. It does not depend on the
  sustain being long, because only its spectrum comes from the recording.
* `<name>_split.sfz` — both as regions on one key, `loop_mode=loop_continuous`.
  Their loop lengths are incommensurate, so the sum never repeats. When
  `pysfizz` is present the SFZ is rendered (`<name>_sfizz.wav`) and checked
  against the tiled sum (level within 0.1 dB, spectrum within ~0.5 dB in every
  audible band).
* `<name>_split_preview.wav` — original | harmonic alone | residual alone |
  both (the sfizz render).

`--harm-bw` decides what "harmonic" means. By default each harmonic takes only
its strongest grid line (the peak-moved main lobe), so a vibrato or a section's
chorus spread stays in the residual as a slowly evolving cluster of lines: a
solo violin then keeps only ~20 % of its energy in the harmonic loop and sounds
chorused. `--harm-bw 30` sums the ±30 cent band around each harmonic into a
steady partial instead (97–98 % of the energy in the harmonic loop, dry, static
tone; vibrato would be re-added by the sampler's LFO).

## Limits, by design

* The loop can only contain frequencies `m·fs/L`. Anything else is moved to the
  nearest grid line (≤ `fs/2L` Hz) — inaudible as pitch, but any vibrato,
  tremolo or beating in the input is turned into a texture that repeats every
  `L` seconds. That is the "assumed constant" in the brief.
* A DCT loop is a palindrome: only `L/2` samples of unique material, with a
  time reversal at the seam and at the middle. For stationary content this is
  inaudible; the `dft` basis is there to A/B it.
* The analysis needs `N ≥ 2L` of steady sustain (`q ≥ 2`; 3–4 is better for the
  noise estimate). The split's residual is exempt: its loop length is free.
* The residual's random phases make it fully decorrelated between channels
  (wide), and a short harmonic loop forces both channels onto one f0, so a
  section's L/R detune (up to 19 cents in SSO) is not reproduced by the split.
