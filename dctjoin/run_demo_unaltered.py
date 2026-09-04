"""Replicate the SSO test set with UNTOUCHED dctloop loops at one or more loop lengths, one
playable .sfz per instrument folder, and a summary table.

    python3 dctjoin/run_demo_unaltered.py [-o dctjoin_out_unaltered] [--loops 0.5,1.5] [--basis dft] [--play]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctjoin.sfz import assign_key_ranges, write_sfz  # noqa: E402
from dctjoin.unaltered import replicate_unaltered, summary_line  # noqa: E402

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
    ap.add_argument('-o', '--out', default='dctjoin_out_unaltered')
    ap.add_argument('--loops', default='0.5,1.5', help='comma-separated loop lengths in seconds')
    ap.add_argument('--basis', default='dft', choices=['dct', 'dft'])
    ap.add_argument('--method', default='bridge', choices=['bridge', 'splice'])
    ap.add_argument('--bridge', type=float, default=0.3, help='bridge length in seconds (method=bridge)')
    ap.add_argument('--xfade', type=float, default=0.01, help='cross-fade length in seconds')
    ap.add_argument('--morph', action='store_true', help='EQ-morph the attack tail (method=splice only)')
    ap.add_argument('--search', type=float, default=0.5, help='seconds to search for the join')
    ap.add_argument('--play', action='store_true')
    ap.add_argument('--files', nargs='*', help='override the test set (paths relative to Samples/)')
    a = ap.parse_args(argv)
    rows = []
    for secs in (float(v) for v in a.loops.split(',')):
        tag = f'{secs:g}s'
        by_inst = defaultdict(list)
        print(f'=== loop {tag} ===')
        for rel in (a.files or TEST_SET):
            inst = os.path.basename(os.path.dirname(rel)).replace(' ', '_')
            out = os.path.join(a.out, tag, inst)
            r = replicate_unaltered(os.path.join(SSO, rel), out, secs, method=a.method, basis=a.basis, morph=a.morph,
                                    search_s=a.search, bridge_s=a.bridge, xfade_s=a.xfade)
            print(summary_line(r))
            rows.append((tag, r))
            by_inst[inst].append(r)
            if a.play and 'preview' in r.outputs:
                subprocess.run(['paplay', r.outputs['preview']])
        for inst, rs in by_inst.items():
            ranges = assign_key_ranges([r.keycenter for r in rs])
            regions = [dict(sample=os.path.basename(r.outputs['audio']), keycenter=r.keycenter, tune_cents=r.tune_cents,
                            loop_start=r.loop_start, loop_end=r.loop_end, lokey=lo, hikey=hi, release=r.release_sfz)
                       for r, (lo, hi) in zip(rs, ranges)]
            write_sfz(os.path.join(a.out, tag, inst, f'{inst}.sfz'), regions,
                      header=f'dctjoin (unaltered {tag} loops): {inst}, {len(rs)} notes')
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'summary.md'), 'w') as fh:
        fh.write('| loop | sample | method | attack s | ncc | harm step dB (rec) | phase err deg (rec) | band step dB (rec) '
                 '| dip dB | untouched | loop s | seam× | wah dB | release s | file s | sfizz |\n')
        fh.write('|' + '---|' * 16 + '\n')
        for tag, r in rows:
            j, m, c = r.junction, r.loop_metrics, r.continuity
            name = os.path.splitext(os.path.basename(r.source))[0]
            s = 'ok' if (r.sfizz and r.sfizz.get('ok')) else ('FAIL' if r.sfizz else 'n/a')
            fh.write(f'| {tag} | {name} | {r.method} | {r.attack_s:.3f} | {r.join_ncc:.2f} | '
                     f'{c["harm_step_db_wmean"]:.2f} ({c["rec_harm_step_db_wmean"]:.2f}) | '
                     f'{c["harm_phase_err_deg_wmean"]:.0f} ({c["rec_harm_phase_err_deg_wmean"]:.0f}) | '
                     f'{c["band_step_db_mean"]:.1f} ({c["rec_band_step_db_mean"]:.1f}) | {j["dip_db"]:+.2f} | '
                     f'{"yes" if r.loop_untouched else "NO"} | {r.loop_seconds:.3f} | {m["seam_flux_ratio"]:.2f} | '
                     f'{m.get("max_harmonic_am_db", 0):.1f} | {r.release_sfz:.2f} | {r.total_seconds:.2f} | {s} |\n')
    print('wrote', os.path.join(a.out, 'summary.md'))


if __name__ == '__main__':
    main()
