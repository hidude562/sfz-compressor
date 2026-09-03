"""dctloop — seamless loops of stationary sounds by reconstruction on a loop-periodic grid.

Give it any number of seconds of a sustained, (assumed) constant sound and get back a loop that
closes by construction: it is rebuilt from cosines whose wavelengths all divide the loop length.

    from dctloop import loop_signal, loop_file

    loop, info = loop_signal(x, fs, seconds=1.5, basis='dct')        # x: (N,) or (N, C) sustain
    result     = loop_file('trumpet-a#4.wav', 'out', seconds=0.25, basis='dft')

``basis='dct'`` gives a palindromic loop (real cosines), ``basis='dft'`` keeps the input's
phases.  Both share the same analysis; see core.py for the method and README.md for the rest.

Lower level: ``make_loop`` (segment + loop length in samples), ``fit_loop_length``,
``analyse_on_grid`` / ``synthesise``, the pitch helpers and the metrics.
"""
from .core import (BASES, analyse_on_grid, fit_loop_length, harmonic_bins, loop_signal, make_loop,
                   synthesise)
from .metrics import harmonic_am, measure, seam_metrics, spectrum_match
from .pipeline import LoopResult, find_body, load_audio, loop_file, make_preview, process, summary_line
from .pitch import (estimate_f0, f0_per_channel, harmonic_evidence, hint_is_trustworthy,
                    note_from_name, refine_f0)
from .split import fit_short_loop, process_split, split_loop

__all__ = [
    'loop_signal', 'loop_file', 'LoopResult', 'BASES',
    'make_loop', 'fit_loop_length', 'harmonic_bins', 'analyse_on_grid', 'synthesise',
    'note_from_name', 'estimate_f0', 'refine_f0', 'f0_per_channel',
    'harmonic_evidence', 'hint_is_trustworthy',
    'seam_metrics', 'harmonic_am', 'spectrum_match', 'measure',
    'load_audio', 'find_body', 'make_preview', 'summary_line', 'process',
    'split_loop', 'process_split', 'fit_short_loop',
]
