"""Loop every genuinely sustained note in the Sonatina Symphonic Orchestra library.

Scope: strings, woodwinds, brass and chorus, played sustain/legato/harmonic (never short
articulations: pizzicato, staccato, spiccato, marcato, tremolo, muted, round-robin hits).
Decay-type "instruments" (piano, mallet and struck/plucked percussion, harp, organ) are out of
scope for a tool whose whole method assumes an (approximately) constant sound — see the SSO
sample survey this script embeds below.

    python3 dctloop/build_library.py [-o out] [--loop 1.5] [--basis dft] [--limit N] [--jobs 1]

Writes <out>/<Instrument>/<name>_loop.wav (+ _preview.wav, .json) for every included file,
mirroring the Samples/ subfolder layout, plus <out>/manifest.md (one row per file: included /
skipped-articulation / skipped-family / failed, with the reason) and <out>/summary.md (the
per-file metrics table, as run_demo.py's).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctloop import loop_file, summary_line  # noqa: E402

SSO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Samples')

# Families whose recordings decay rather than sustain: this tool rebuilds a sound from the
# frequencies that fit exactly in the loop, which only makes sense for a tone that is
# (approximately) constant over the analysis window.  A piano or mallet note has no such
# window; looping one is a different, legitimate technique (extend a decaying tail) but not
# what dctloop does, so these are out of scope here rather than silently mis-processed.
DECAY_FAMILIES = {'grand piano', 'marimba', 'crotales', 'vibraphone', 'glockenspiel', 'chimes',
                  'harp', 'xylophone', 'percussion', 'harpsichord', 'organ', 'celeste'}

# Articulation tokens (SSO's file-naming convention splits them with '-' or '_') that mark a
# short, non-sustained sample: attack-only or decay-only, not a steady body to analyse.
SKIP_TOKENS = {'piz', 'pizz', 'stc', 'stacc', 'stac', 'spic', 'spicc', 'spiccato', 'marc',
               'marcato', 'trm', 'trem', 'tremolo', 'legno', 'sord', 'mute', 'muted',
               'rr1', 'rr2', 'rr3'}

_NOTE_RE = re.compile(r'-?[a-g]#?-?\d+', re.I)


def _articulation_tokens(stem: str) -> list[str]:
    body = _NOTE_RE.sub('', stem)
    return [t for t in re.split(r'[-_]', body) if t]


def find_sustain_files(root: str = SSO) -> tuple[list[str], list[tuple[str, str]]]:
    """(included paths, [(path, reason)] for every file left out), reasons are 'family' or
    'articulation'.  Only real directories are scanned; SSO's case-variant folders
    ('trumpet' next to 'Trumpet') are symlinks to one another."""
    included, excluded = [], []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full) or os.path.islink(full):
            continue
        family_out = d.lower() in DECAY_FAMILIES
        for f in sorted(os.listdir(full)):
            if not f.lower().endswith(('.wav', '.flac', '.aif', '.aiff')):
                continue
            path = os.path.join(full, f)
            if family_out:
                excluded.append((path, 'family'))
                continue
            stem = os.path.splitext(f)[0]
            if any(t in SKIP_TOKENS for t in _articulation_tokens(stem)):
                excluded.append((path, 'articulation'))
                continue
            included.append(path)
    return included, excluded


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', default='out')
    ap.add_argument('--loop', type=float, default=1.5)
    ap.add_argument('--basis', default='dft', choices=['dct', 'dft'])
    ap.add_argument('--limit', type=int, help='stop after this many files (smoke-testing)')
    ap.add_argument('--pyin', action='store_true', help='force pYIN instead of the file-name pitch')
    a = ap.parse_args(argv)

    included, excluded = find_sustain_files()
    if a.limit:
        included = included[:a.limit]
    print(f'{len(included)} sustained files to loop, {len(excluded)} skipped '
          f'({sum(1 for _, r in excluded if r == "family")} decay-family, '
          f'{sum(1 for _, r in excluded if r == "articulation")} short articulation)')

    os.makedirs(a.out, exist_ok=True)
    manifest = [('file', 'status', 'detail')]
    for p, reason in excluded:
        manifest.append((os.path.relpath(p, SSO), 'skipped', reason))

    rows, t0 = [], time.time()
    for i, path in enumerate(included, 1):
        instrument = os.path.relpath(os.path.dirname(path), SSO)
        out_dir = os.path.join(a.out, instrument)
        rel = os.path.relpath(path, SSO)
        try:
            r = loop_file(path, out_dir, a.loop, basis=a.basis, use_hint=not a.pyin)
            rows.append(r)
            manifest.append((rel, 'ok', r.info.get('pitch', {}).get('method', '?')))
        except Exception as e:
            manifest.append((rel, 'failed', str(e)))
        if i % 50 == 0 or i == len(included):
            dt = time.time() - t0
            print(f'  {i:4d}/{len(included)}  ({dt:.0f}s, {dt / i * 1000:.0f} ms/file avg)')

    with open(os.path.join(a.out, 'manifest.md'), 'w') as fh:
        fh.write('| file | status | detail |\n|---|---|---|\n')
        for row in manifest[1:]:
            fh.write('| ' + ' | '.join(row) + ' |\n')

    with open(os.path.join(a.out, 'summary.md'), 'w') as fh:
        fh.write('| file | basis | f0 Hz | pitch | L/R detune c | loop s | periods | seam× | wah dB '
                 '| LTAS mean/max dB | mono dB |\n')
        fh.write('|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            m = r.metrics
            rel = os.path.relpath(r.source, SSO)
            pm = r.info.get('pitch', {}).get('method', '?')
            fh.write(f'| {rel} | {r.basis} | {r.f0_used:.2f} | {pm} | {r.detune_cents:+.1f} | '
                     f'{r.loop_seconds:.3f} | {r.K} | {m["seam_flux_ratio"]:.2f} | '
                     f'{m.get("max_harmonic_am_db", 0):.1f} | {m["ltas_mean_abs_db"]:.2f}/{m["ltas_max_abs_db"]:.2f} '
                     f'| {m["mono_db"]:+.1f} |\n')

    failed = [row for row in manifest[1:] if row[1] == 'failed']
    print(f'\n{len(rows)} loops built, {len(failed)} failed, {sum(1 for _, s, _ in manifest[1:] if s == "skipped")} skipped '
          f'in {time.time() - t0:.1f}s')
    print('wrote', os.path.join(a.out, 'summary.md'), 'and', os.path.join(a.out, 'manifest.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
