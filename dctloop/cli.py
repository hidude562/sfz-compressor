"""dctloop command line.

    python3 -m dctloop note.wav [note2.wav ... | directory] -o out --loop 1.5 --basis dct [--play]

Writes <name>_loop.wav, <name>_preview.wav (original, gap, loop repeated) and <name>.json per
input.  ``--split`` switches to the experimental harmonic + residual two-loop output (split.py).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .pipeline import loop_file
from .split import process_split


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='dctloop', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='wav/flac files (or directories) of sustained notes')
    p.add_argument('-o', '--out', default='dctloop_out', help='output directory (default dctloop_out)')
    p.add_argument('--loop', type=float, default=1.5, help='target loop length in seconds (default 1.5)')
    p.add_argument('--periods', type=int, help='instead of --loop: exactly this many f0 periods')
    p.add_argument('--basis', choices=['dct', 'dft'], default='dct',
                   help='dct: real cosines, palindromic loop (default); dft: original phases, no mirror')
    p.add_argument('--phase', choices=['orig', 'random'], default='orig',
                   help='partial phases (dct: signs) from the input, or random')
    p.add_argument('--f0', type=float, help='fundamental in Hz (default: note in the file name + pYIN, refined)')
    p.add_argument('--no-fit', action='store_true', help='do not round the loop to an integer number of periods')
    p.add_argument('--lock', type=float, default=1.5, metavar='BINS',
                   help='harmonic locking half-width in grid bins (default 1.5; 0 disables)')
    p.add_argument('--start', type=float, help='analysis segment start (s); default: auto sustain detection')
    p.add_argument('--dur', type=float, help='analysis segment length (s); default: whole sustain')
    p.add_argument('--no-preview', action='store_true', help='skip the original+loop preview file')
    p.add_argument('--play', action='store_true', help='play each preview with paplay after building it')
    p.add_argument('--seed', type=int, default=0, help='for --phase random')
    p.add_argument('-v', '--verbose', action='store_true', help='re-raise errors instead of skipping the file')
    g = p.add_argument_group('split mode (experimental, dctloop/split.py)')
    g.add_argument('--split', action='store_true',
                   help='harmonic loop (a few periods) + long residual loop + two-region SFZ instead of one loop')
    g.add_argument('--resid-loop', type=float, default=3.0, help='residual loop length in seconds (default 3)')
    g.add_argument('--harm-periods', type=int, help='periods in the harmonic loop (default: shortest within 0.5 cent)')
    g.add_argument('--harm-bw', type=float, help='sum +-this many cents around each harmonic into it')
    g.add_argument('--no-sfizz', action='store_true', help='skip the sfizz render check')
    return p


def _expand(inputs):
    exts = ('.wav', '.flac', '.aif', '.aiff', '.ogg')
    for item in inputs:
        if os.path.isdir(item):
            for n in sorted(os.listdir(item)):
                if n.lower().endswith(exts):
                    yield os.path.join(item, n)
        else:
            yield item


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    for path in _expand(a.inputs):
        try:
            if a.split:
                r = process_split(path, a.out, resid_seconds=a.resid_loop, harm_periods=a.harm_periods,
                                  harm_bw_cents=a.harm_bw, f0=a.f0, start=a.start, dur=a.dur, seed=a.seed,
                                  verbose=True, sfizz=not a.no_sfizz)
            else:
                r = loop_file(path, a.out, a.loop, basis=a.basis, f0=a.f0, fit=not a.no_fit,
                              periods=a.periods, lock=a.lock, phase=a.phase, start=a.start, dur=a.dur,
                              preview=not a.no_preview, seed=a.seed, verbose=True)
        except Exception as e:  # keep going through a directory
            print(f'  [{os.path.basename(path)}] FAILED: {e}', file=sys.stderr)
            if a.verbose:
                raise
            continue
        if a.play and 'preview' in r.outputs:
            subprocess.run(['paplay', r.outputs['preview']])
    return 0
