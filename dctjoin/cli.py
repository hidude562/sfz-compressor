"""dctjoin command line.

    python3 -m dctjoin note.wav [note2.wav ... | directory] -o out --loop 0.25 --basis dft [--play]

Per input writes <name>.flac (attack + loop), <name>.sfz, <name>_preview.wav (original, gap,
replication), <name>_sfizz.wav (the .sfz played by sfizz, if pysfizz is installed) and <name>.json.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .pipeline import replicate_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='dctjoin', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='wav/flac files (or directories) of recorded notes')
    p.add_argument('-o', '--out', default='dctjoin_out', help='output directory (default dctjoin_out)')
    p.add_argument('--loop', type=float, default=0.25, help='target loop length in seconds (default 0.25)')
    p.add_argument('--basis', choices=['dct', 'dft'], default='dft',
                   help='dft (default): phases can be aligned to the attack; dct: rotation only')
    p.add_argument('--align', choices=['best', 'phase', 'rotate', 'none'], default='best',
                   help='how to line the loop up with the recording at the join (default: try both, keep the better)')
    p.add_argument('--f0', type=float, help='fundamental in Hz (default: file name, verified; else pYIN)')
    p.add_argument('--pyin', action='store_true', help='always run pYIN instead of trusting the file name')
    p.add_argument('--lock', type=float, default=1.5, help='dctloop harmonic-lock width in grid bins')
    p.add_argument('--xfade', type=float, default=0.01, help='cross-fade length in seconds (default 0.01, at least one period)')
    p.add_argument('--no-morph', action='store_true', help='do not EQ-morph the attack tail towards the loop')
    p.add_argument('--morph-s', type=float, default=0.25, help='length of the morphed tail (default 0.25)')
    p.add_argument('--morph-db', type=float, default=6.0, help='maximum morph gain per band (default 6)')
    p.add_argument('--joins', default='0,0.15,0.3',
                   help='join candidates as seconds after the attack end (default 0,0.15,0.3); the best junction wins')
    p.add_argument('--format', choices=['flac', 'wav'], default='flac')
    p.add_argument('--release', type=float, default=0.25, help='ampeg_release written to the sfz')
    p.add_argument('--no-preview', action='store_true')
    p.add_argument('--no-sfizz', action='store_true', help='skip the sfizz render check')
    p.add_argument('--play', action='store_true', help='play each preview with paplay')
    p.add_argument('-v', '--verbose', action='store_true', help='re-raise errors instead of skipping the file')
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
    joins = tuple(float(v) for v in a.joins.split(','))
    for path in _expand(a.inputs):
        try:
            r = replicate_file(path, a.out, a.loop, basis=a.basis, f0=a.f0, use_hint=not a.pyin, lock=a.lock,
                               align=a.align, xfade_s=a.xfade, morph=not a.no_morph, morph_s=a.morph_s,
                               morph_max_db=a.morph_db, join_offsets=joins, format=a.format,
                               preview=not a.no_preview, sfizz=not a.no_sfizz, release=a.release, verbose=True)
        except Exception as e:
            print(f'  [{os.path.basename(path)}] FAILED: {e}', file=sys.stderr)
            if a.verbose:
                raise
            continue
        if a.play and 'preview' in r.outputs:
            subprocess.run(['paplay', r.outputs['preview']])
    return 0
