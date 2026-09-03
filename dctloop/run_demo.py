"""Run dctloop over the SSO test set (trumpets, solo strings, string sections) with each basis
and write a summary table.

    python3 dctloop/run_demo.py [-o dctloop_out] [--loop 1.5] [--basis dct,dft] [--play]
    python3 dctloop/run_demo.py --split          # the experimental two-loop output instead
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctloop import loop_file, process_split, summary_line  # noqa: E402

SSO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Samples')
TEST_SET = [
    'trumpet/trumpet-a#4.wav',
    'trumpet/trumpet-g3.wav',
    'trumpet/trumpet-c#5.wav',
    'violin/violin-a#4.wav',
    'cello/cello-c3.wav',
    '1st violins/1st-violins-sus-a#4.wav',
    '2nd violins/2nd-violins-sus-e4.wav',
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', default='dctloop_out')
    ap.add_argument('--loop', type=float, default=1.5)
    ap.add_argument('--basis', default='dct,dft', help='comma-separated: dct, dft')
    ap.add_argument('--lock', type=float, default=1.5)
    ap.add_argument('--play', action='store_true')
    ap.add_argument('--files', nargs='*', help='override the test set (paths relative to Samples/)')
    ap.add_argument('--split', action='store_true', help='harmonic + residual split instead of single loops')
    ap.add_argument('--resid-loop', type=float, default=3.0)
    ap.add_argument('--harm-bw', type=float)
    a = ap.parse_args(argv)
    if a.split:
        out = os.path.join(a.out, 'split')
        for rel in (a.files or TEST_SET):
            r = process_split(os.path.join(SSO, rel), out, resid_seconds=a.resid_loop, harm_bw_cents=a.harm_bw, verbose=True)
            if a.play:
                subprocess.run(['paplay', r.outputs['preview']])
        return
    rows = []
    for rel in (a.files or TEST_SET):
        path = os.path.join(SSO, rel)
        for basis in a.basis.split(','):
            r = loop_file(path, os.path.join(a.out, basis), a.loop, basis=basis, lock=a.lock)
            rows.append(r)
            print(summary_line(r))
            if a.play and 'preview' in r.outputs:
                subprocess.run(['paplay', r.outputs['preview']])
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'summary.md'), 'w') as fh:
        fh.write('| sample | basis | f0 Hz | L/R detune c | loop s | periods | seam× | mid× | p95× '
                 '| LTAS mean/max dB | mono dB | wah dB | gain |\n')
        fh.write('|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            m = r.metrics
            name = os.path.splitext(os.path.basename(r.source))[0]
            fh.write(f'| {name} | {r.basis} | {r.f0_used:.2f} | {r.detune_cents:+.1f} | {r.loop_seconds:.3f} | {r.K} | '
                     f'{m["seam_flux_ratio"]:.2f} | {m["mid_flux_ratio"]:.2f} | {m["p95_flux_ratio"]:.2f} | '
                     f'{m["ltas_mean_abs_db"]:.2f}/{m["ltas_max_abs_db"]:.2f} | {m["mono_db"]:+.1f} | '
                     f'{m.get("max_harmonic_am_db", 0):.1f} | {r.info["gain"]:.2f} |\n')
    print('wrote', os.path.join(a.out, 'summary.md'))


if __name__ == '__main__':
    main()
