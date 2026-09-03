"""dctloop command line.

    python3 -m dctloop note.wav [note2.wav ...] -o out --loop 1.5 --mode snap --basis dct [--play]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .core import process


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='dctloop', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='wav/flac files (or directories) of sustained notes')
    p.add_argument('-o', '--out', default='dctloop_out', help='output directory')
    p.add_argument('--loop', type=float, default=1.5, help='target loop length in seconds (default 1.5)')
    p.add_argument('--periods', type=int, help='instead of --loop: exactly this many f0 periods')
    p.add_argument('--mode', choices=['snap', 'comb'], default='snap',
                   help="snap: Welch energy on the loop grid (keeps noise); comb: one frame, attenuates off-grid content")
    p.add_argument('--basis', choices=['dct', 'dft'], default='dct',
                   help='dct: real cosines, palindromic loop (default); dft: complex, original phases')
    p.add_argument('--phase', choices=['orig', 'random'], default='orig',
                   help="partial phases (dct: signs) from the input, or random")
    p.add_argument('--f0', type=float, help='fundamental in Hz (default: parse the note from the file name + pYIN)')
    p.add_argument('--no-fit', action='store_true', help='do not round the loop to an integer number of periods')
    p.add_argument('--start', type=float, help='analysis segment start (s); default: auto sustain detection')
    p.add_argument('--dur', type=float, help='analysis segment length (s); default: whole sustain')
    p.add_argument('--no-preview', action='store_true', help='skip the original+loop preview file')
    p.add_argument('--play', action='store_true', help='play each preview with paplay after building it')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-v', '--verbose', action='store_true')
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
            r = process(path, a.out, loop=a.loop, mode=a.mode, basis=a.basis, phase=a.phase,
                        f0=a.f0, fit=not a.no_fit, start=a.start, dur=a.dur,
                        periods=a.periods, preview=not a.no_preview, seed=a.seed, verbose=True)
        except Exception as e:  # keep going through a directory
            print(f'  [{os.path.basename(path)}] FAILED: {e}', file=sys.stderr)
            if a.verbose:
                raise
            continue
        if a.play and 'preview' in r.outputs:
            subprocess.run(['paplay', r.outputs['preview']])
    return 0
