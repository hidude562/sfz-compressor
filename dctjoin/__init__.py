"""dctjoin — the attack of a recorded note, joined seamlessly to a dctloop loop.

Sister library to dctloop.  dctloop makes a loop of the steady part of a note; dctjoin keeps the
recorded attack as it is and hands over to that loop without an audible step: the loop is
phase-aligned to the recording at the join (DFT basis), level-matched there, the attack's tail is
EQ-morphed towards the loop's timbre, and the two are spliced with a short coherent cross-fade.
The result is one audio file with SFZ loop points, plus an .sfz that plays it.

    from dctjoin import replicate_file, join

    r = replicate_file('Samples/trumpet/trumpet-a#4.wav', 'out', seconds=0.25, basis='dft')
    r.outputs['sfz']                     # out/trumpet-a#4.sfz  (+ .flac, _preview.wav, _sfizz.wav, .json)

    out, info = join(x, fs, onset, join_at, loop, f0)     # arrays in, (audio, loop points) out
"""
from .join import (best_rotation, join, junction_metrics, level_match, morph_tail, phase_align_loop,
                   rotate_loop, splice)
from .pipeline import Replication, make_preview, render_sfizz, replicate_file, summary_line
from .segment import Segments, segment_note
from .sfz import assign_key_ranges, key_and_tune, midi_from_hz, write_sfz

__all__ = ['replicate_file', 'Replication', 'join', 'phase_align_loop', 'rotate_loop', 'best_rotation',
           'level_match', 'morph_tail', 'splice', 'junction_metrics', 'segment_note', 'Segments',
           'key_and_tune', 'midi_from_hz', 'assign_key_ranges', 'write_sfz', 'make_preview',
           'render_sfizz', 'summary_line']
