"""Replicate the SSO test set (trumpets, solo strings, string sections) and build one playable
.sfz instrument per instrument folder.

    python3 dctjoin/run_demo.py [-o dctjoin_out] [--loop 0.25] [--basis dft] [--play]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctjoin import assign_key_ranges, replicate_file, summary_line, write_sfz  # noqa: E402

SSO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Samples')
TEST_SET = [
    'trumpet/trumpet-g3.wav',
    'trumpet/trumpet-a#4.wav',
    'trumpet/trumpet-c#5.wav',
    'violin/violin-a#4.wav',
    'cello/cello-c3.wav',
    '1st violins/1st-violins-sus-a#4.wav',
    '2nd violins/2nd-violins-sus-e4.wav',
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', default='dctjoin_out')
    ap.add_argument('--loop', type=float, default=0.25)
    ap.add_argument('--basis', default='dft', choices=['dct', 'dft'])
    ap.add_argument('--align', default='best', choices=['best', 'phase', 'rotate', 'none'])
    ap.add_argument('--no-morph', action='store_true')
    ap.add_argument('--joins', default='0,0.15,0.3')
    ap.add_argument('--play', action='store_true')
    ap.add_argument('--files', nargs='*', help='override the test set (paths relative to Samples/)')
    a = ap.parse_args(argv)
    joins = tuple(float(v) for v in a.joins.split(','))
    rows, by_inst = [], defaultdict(list)
    for rel in (a.files or TEST_SET):
        inst = os.path.basename(os.path.dirname(rel)).replace(' ', '_')
        out = os.path.join(a.out, inst)
        r = replicate_file(os.path.join(SSO, rel), out, a.loop, basis=a.basis, align=a.align,
                           morph=not a.no_morph, join_offsets=joins)
        print(summary_line(r))
        rows.append(r)
        by_inst[inst].append(r)
        if a.play and 'preview' in r.outputs:
            subprocess.run(['paplay', r.outputs['preview']])
    # one instrument .sfz per folder, key ranges split between the notes
    for inst, rs in by_inst.items():
        ranges = assign_key_ranges([r.keycenter for r in rs])
        regions = [dict(sample=os.path.basename(r.outputs['audio']), keycenter=r.keycenter, tune_cents=r.tune_cents,
                        loop_start=r.loop_start, loop_end=r.loop_end, lokey=lo, hikey=hi)
                   for r, (lo, hi) in zip(rs, ranges)]
        write_sfz(os.path.join(a.out, inst, f'{inst}.sfz'), regions, header=f'dctjoin: {inst}, {len(rs)} notes')
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'summary.md'), 'w') as fh:
        fh.write('| sample | f0 Hz | key | attack s | join s | loop s | align | dip dB | transient dB | score '
                 '| loop seam× | wah dB | LTAS dB | sfizz |\n')
        fh.write('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            j, m = r.junction, r.loop_metrics
            name = os.path.splitext(os.path.basename(r.source))[0]
            s = 'ok' if (r.sfizz and r.sfizz.get('ok')) else ('FAIL' if r.sfizz else 'n/a')
            fh.write(f'| {name} | {r.f0_used:.2f} | {r.keycenter}{r.tune_cents:+.0f}c | '
                     f'{r.segments_s["attack_end"] - r.segments_s["onset"]:.3f} | {r.join_s - r.segments_s["onset"]:.3f} | '
                     f'{r.loop_seconds:.3f} | {r.alignment} | {j["dip_db"]:+.2f} | {j["transient_db"]:+.1f} | {j["score"]:.2f} | '
                     f'{m["seam_flux_ratio"]:.2f} | {m.get("max_harmonic_am_db", 0):.1f} | {m["ltas_mean_abs_db"]:.2f} | {s} |\n')
    print('wrote', os.path.join(a.out, 'summary.md'))


if __name__ == '__main__':
    main()
