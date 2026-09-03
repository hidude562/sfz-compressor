"""Run dctloop over the SSO test set (trumpets, strings) in every basis/mode combination and
write a summary table.  Usage: python3 dctloop/run_demo.py [-o dctloop_out] [--loop 1.5] [--play]"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctloop.core import process  # noqa: E402

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
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='dctloop_out')
    ap.add_argument('--loop', type=float, default=1.5)
    ap.add_argument('--configs', default='dct-snap,dct-comb,dft-snap,dft-comb')
    ap.add_argument('--play', action='store_true')
    ap.add_argument('--files', nargs='*', help='override the test set (paths relative to Samples/)')
    a = ap.parse_args(argv)
    rows = []
    for rel in (a.files or TEST_SET):
        path = os.path.join(SSO, rel)
        for cfg in a.configs.split(','):
            basis, mode = cfg.split('-')
            out = os.path.join(a.out, cfg)
            r = process(path, out, loop=a.loop, mode=mode, basis=basis, verbose=False)
            rows.append((os.path.splitext(os.path.basename(rel))[0], cfg, r))
            print(f'{rows[-1][0]:32s} {cfg:9s} f0={r.f0_used:7.2f}Hz det={r.detune_cents:+5.1f}c '
                  f'L={r.loop_seconds:.3f}s K={r.K:4d} q={r.info["q"]} '
                  f'seam×{r.seam["seam_flux_ratio"]:.2f} mid×{r.seam["mid_flux_ratio"]:.2f} '
                  f'p95×{r.seam["p95_flux_ratio"]:.2f} ltas={r.spectrum["ltas_mean_abs_db"]:.2f}/'
                  f'{r.spectrum["ltas_max_abs_db"]:.2f}dB mono={r.spectrum["mono_db"]:+.1f}dB gain={r.info["gain"]:.2f} grid-off={r.info.get("grid_offset")}')
            if a.play and 'preview' in r.outputs:
                subprocess.run(['paplay', r.outputs['preview']])
    with open(os.path.join(a.out, 'summary.md'), 'w') as fh:
        fh.write('| sample | config | f0 Hz | L/R detune c | loop s | periods | seam× | mid× | p95× | LTAS mean/max dB | mono dB | gain |\n')
        fh.write('|---|---|---|---|---|---|---|---|---|---|---|---|\n')
        for name, cfg, r in rows:
            fh.write(f'| {name} | {cfg} | {r.f0_used:.2f} | {r.detune_cents:+.1f} | {r.loop_seconds:.3f} | {r.K} | '
                     f'{r.seam["seam_flux_ratio"]:.2f} | {r.seam["mid_flux_ratio"]:.2f} | {r.seam["p95_flux_ratio"]:.2f} | '
                     f'{r.spectrum["ltas_mean_abs_db"]:.2f}/{r.spectrum["ltas_max_abs_db"]:.2f} | {r.spectrum["mono_db"]:+.1f} | {r.info["gain"]:.2f} |\n')
    print('wrote', os.path.join(a.out, 'summary.md'))


if __name__ == '__main__':
    main()
