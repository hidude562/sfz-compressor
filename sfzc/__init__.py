"""sfzc - automatic SFZ sample looping by harmonic-plus-noise resynthesis.

Algorithm A (``sfzc.looper``): length-constrained resynthesis of a sampled note
into an exactly periodic loop (per-partial phase/amplitude matched, Laroche
US 6,084,170 / 6,239,345 style) plus a seamless stochastic residual, with the
amplitude/brightness evolution expressed as standard SFZ envelope opcodes.

Algorithm B (``sfzc.metric``): a five-term perceptual recreation metric
(spectral envelope, loudness envelope, seam detectability, temporal variation,
pitch/vibrato consistency) that scores a rendered SFZ against the original.
"""
__version__ = "0.1.0"
