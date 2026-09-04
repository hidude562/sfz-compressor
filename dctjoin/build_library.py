"""Replicate every sustained note of the Sonatina Symphonic Orchestra as recorded attack +
untouched dctloop loop, and write one playable .sfz per instrument variant.

    python3 dctjoin/build_library.py [-o dctjoin_library] [--loop 0.5] [--max-attack 0.5]
                                     [--folders trumpet,horns] [--limit N] [--no-sfizz]
    python3 dctjoin/build_library.py -o dctjoin_library --combine-only     # just the .sfz files

Selection: string, woodwind, brass and chorus folders; sustained articulations only (sus,
plain notes, harmonics, arco vib / non-vib).  Short articulations (pizzicato, staccato,
spiccato, tremolo, col legno, round-robin hits) and decaying families (keyboards, mallets,
harp, percussion, organ stops) are skipped and listed in the manifest.

Grouping: files in a folder are grouped by *variant* (every token in the name except the note
and the dynamic: 'trumpet', '1st-violins-sus', '1st-violins-hrm', 'fl2_sus_vb', 'horns-sus')
and, within a variant, by *dynamic* (pp p mp mf f ff -> velocity layers).  Each variant becomes
<out>/<Folder>/<variant>.sfz with contiguous key ranges split at the midpoints between the
notes actually present and velocity ranges split evenly between the dynamics present, and each
folder becomes <out>/<Folder>/<Folder>.sfz holding every variant: one plays as it is, several
are keyswitched (keys below the playing range, the sustain articulation as default).

Per note the replication is dctjoin.unaltered.replicate_unaltered (method='bridge', loop
untouched, attack <= --max-attack); the loop is exactly dctloop's.  <out>/manifest.md lists
every file with its status, <out>/summary.md the continuity numbers per replicated note.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctloop import note_from_name  # noqa: E402
from dctjoin.unaltered import replicate_unaltered  # noqa: E402

SSO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Samples')

DECAY_FAMILIES = {'grand piano', 'marimba', 'crotales', 'vibraphone', 'glockenspiel', 'chimes', 'harp',
                  'xylophone', 'percussion', 'harpsichord', 'organ', 'celeste'}
SKIP_TOKENS = {'piz', 'pizz', 'stc', 'stacc', 'stac', 'spic', 'spicc', 'spiccato', 'marc', 'marcato', 'trm',
               'trem', 'tremolo', 'col', 'legno', 'sord', 'mute', 'muted', 'roll', 'mech', 'rr1', 'rr2', 'rr3'}
IGNORE_TOKENS = {'pb', 'loop'}                      # SSO's pre-looped horns: 'horns-sus-ff-a#2-PB-loop'
DYNAMICS = ['pp', 'p', 'mp', 'mf', 'f', 'ff']
_NOTE = re.compile(r'^[a-g](#|b)?-?\d$', re.I)


def parse_name(stem: str) -> dict | None:
    """{'variant', 'dynamic', 'note'} from an SSO file stem, or None if there is no note token."""
    toks = [t for t in re.split(r'[-_ ]', stem) if t]
    note_i = [i for i, t in enumerate(toks) if _NOTE.match(t)]
    if not note_i:
        return None
    ni = note_i[-1]
    rest = [t for i, t in enumerate(toks) if i != ni and t.lower() not in IGNORE_TOKENS]
    dyn = [t.lower() for t in rest if t.lower() in DYNAMICS]
    variant = [t for t in rest if t.lower() not in DYNAMICS]
    return dict(variant='_'.join(variant).lower() or 'default', dynamic=(dyn[-1] if dyn else None), note=toks[ni],
                skip=any(t.lower() in SKIP_TOKENS for t in rest))


def find_sustains(root: str = SSO, folders: set | None = None):
    """(included: [(path, folder, variant, dynamic)], excluded: [(relpath, reason)])"""
    inc, exc = [], []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full) or os.path.islink(full):
            continue
        if folders and d.lower() not in folders:
            continue
        for f in sorted(os.listdir(full)):
            if not f.lower().endswith(('.wav', '.flac', '.aif', '.aiff')):
                continue
            rel = os.path.join(d, f)
            if d.lower() in DECAY_FAMILIES:
                exc.append((rel, 'decay family'))
                continue
            info = parse_name(os.path.splitext(f)[0])
            if info is None:
                exc.append((rel, 'no note in the name'))
                continue
            if info['skip']:
                exc.append((rel, 'short articulation'))
                continue
            inc.append((os.path.join(full, f), d, info['variant'], info['dynamic']))
    return inc, exc


def key_ranges(keys: list[int], max_stretch: int = 7) -> list[tuple[int, int]]:
    """Contiguous lokey/hikey per note, split at the midpoints between neighbours."""
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ks = [keys[i] for i in order]
    out = [None] * len(ks)
    for j, k in enumerate(ks):
        lo = max(0, k - max_stretch) if j == 0 else max(k - max_stretch, (ks[j - 1] + k) // 2 + 1)
        hi = min(127, k + max_stretch) if j == len(ks) - 1 else min(k + max_stretch, (k + ks[j + 1]) // 2)
        out[order[j]] = (int(lo), int(max(lo, hi)))
    return out


def vel_ranges(dynamics: list) -> dict:
    """Velocity ranges per dynamic present, in pp..ff order, splitting 1..127 evenly."""
    present = [d for d in DYNAMICS if d in dynamics]
    if not present:
        return {None: (1, 127)}
    n = len(present)
    edges = [1 + round(i * 126 / n) for i in range(n)] + [127]
    return {d: (edges[i] if i == 0 else edges[i] + 1, edges[i + 1]) for i, d in enumerate(present)}


def write_variant_sfz(path: str, variant: str, folder: str, reps: list) -> int:
    """reps: list of (Replication, dynamic).  One region per note per dynamic."""
    vr = vel_ranges([d for _, d in reps])
    lines = [f'// dctjoin replication: {folder} / {variant}, {len(reps)} notes, '
             f'{len(vr)} velocity layer(s).  Recorded attack -> untouched dctloop loop.',
             '<control>', 'default_path=', '<global>', 'loop_mode=loop_continuous']
    rl, n = region_lines(reps)
    lines += rl
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return n


KEYSWITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _note_name(k: int) -> str:
    return f'{KEYSWITCH_NAMES[k % 12]}{k // 12 - 1}'


def _default_variant(variants: list[str]) -> str:
    """The articulation that plays with no keyswitch: 'sus' if there is one, else the plain
    instrument name (no articulation token), else the first alphabetically."""
    for v in variants:
        if v.endswith('_sus') or '_sus_' in v:
            return v
    plain = [v for v in variants if not any(t in v for t in ('hrm', 'vib', 'nv'))]
    return plain[0] if plain else variants[0]


def region_lines(reps: list, sw_last: int | None = None) -> tuple[list[str], int]:
    """<region> lines for one variant: reps = [(rep, dynamic)], contiguous key ranges per dynamic,
    velocity ranges from the dynamics present, an optional keyswitch.  Returns (lines, n_regions)."""
    vr = vel_ranges([d for _, d in reps])
    by_dyn = defaultdict(list)
    for r, dyn in reps:
        by_dyn[dyn].append(r)
    lines, n = [], 0
    for dyn, rs in by_dyn.items():
        seen, uniq = set(), []
        for r in sorted(rs, key=lambda r: r.keycenter):
            if r.keycenter in seen:
                continue
            seen.add(r.keycenter)
            uniq.append(r)
        lo_v, hi_v = vr.get(dyn, (1, 127))
        for r, (lo, hi) in zip(uniq, key_ranges([r.keycenter for r in uniq])):
            sw = f' sw_last={sw_last}' if sw_last is not None else ''
            lines.append(f'<region> sample={os.path.basename(r.outputs["audio"])} pitch_keycenter={r.keycenter} '
                         f'lokey={lo} hikey={hi} lovel={lo_v} hivel={hi_v} tune={int(round(-r.tune_cents))} '
                         f'loop_start={r.loop_start} loop_end={r.loop_end} ampeg_release={r.release_sfz:.3f}{sw}')
            n += 1
    return lines, n


def write_instrument_sfz(path: str, folder: str, groups: dict) -> dict:
    """One .sfz for a whole instrument folder: groups = {variant: [(rep, dynamic)]}.  A single
    variant plays as it is; several are keyswitched (sw_lokey..sw_hikey below the playing range,
    sw_default = the sustain articulation), one <group> per variant."""
    variants = sorted(groups)
    info = dict(variants=variants, regions=0, keyswitches={})
    lines = [f'// dctjoin replication: {folder} - {len(variants)} articulation(s), recorded attack -> untouched dctloop loop.',
             '<control>', 'default_path=']
    if len(variants) == 1:
        lines += ['<global>', 'loop_mode=loop_continuous']
        rl, n = region_lines(groups[variants[0]])
        lines += rl
        info['regions'] = n
    else:
        lowest = min(kr[0] for reps in groups.values() for kr in key_ranges([r.keycenter for r, _ in reps]))
        ks0 = 12 if lowest > 12 + len(variants) else 0
        default = _default_variant(variants)
        ks = {v: ks0 + i for i, v in enumerate(variants)}
        lines[0] += (f'  Keyswitches {_note_name(ks0)}..{_note_name(ks0 + len(variants) - 1)} '
                     f'(MIDI {ks0}..{ks0 + len(variants) - 1}); default {default}.')
        for v in variants:
            lines.append(f'//   {_note_name(ks[v])} ({ks[v]:3d}): {v}')
        lines += ['<global>', f'loop_mode=loop_continuous sw_lokey={ks0} sw_hikey={ks0 + len(variants) - 1} sw_default={ks[default]}']
        for v in variants:
            lines.append(f'<group> sw_last={ks[v]} sw_label={v}')
            rl, n = region_lines(groups[v], sw_last=None)
            lines += rl
            info['regions'] += n
        info['keyswitches'] = ks
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return info


def combine(out_dir: str, remove_note_sfz: bool = True) -> list:
    """Build every <Folder>/<Folder>.sfz (and refresh the per-variant .sfz files) from the
    per-note .json files already in ``out_dir``; optionally delete the per-note .sfz files."""
    import json
    from types import SimpleNamespace
    results = []
    for folder in sorted(os.listdir(out_dir)):
        fdir = os.path.join(out_dir, folder)
        if not os.path.isdir(fdir):
            continue
        groups = defaultdict(list)
        for jp in sorted(glob_json(fdir)):
            j = json.load(open(jp))
            info = parse_name(os.path.splitext(os.path.basename(j['source']))[0])
            if info is None:
                continue
            rep = SimpleNamespace(keycenter=j['keycenter'], tune_cents=j['tune_cents'], loop_start=j['loop_start'],
                                  loop_end=j['loop_end'], release_sfz=j['release_sfz'], outputs=j['outputs'])
            groups[info['variant']].append((rep, info['dynamic']))
        if not groups:
            continue
        if remove_note_sfz:
            for sp in os.listdir(fdir):
                if sp.endswith('.sfz'):
                    with open(os.path.join(fdir, sp)) as fh:
                        first = fh.readline()
                    if first.startswith('// dctjoin (unaltered loop) replication of') or first.startswith('// temporary'):
                        os.remove(os.path.join(fdir, sp))
        for v, reps in groups.items():
            write_variant_sfz(os.path.join(fdir, f'{v}.sfz'), v, folder, reps)
        inst = write_instrument_sfz(os.path.join(fdir, f'{folder}.sfz'), folder, groups)
        results.append((folder, inst))
    return results


def glob_json(fdir: str) -> list:
    return [os.path.join(fdir, f) for f in os.listdir(fdir) if f.endswith('.json')]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', default='dctjoin_library')
    ap.add_argument('--loop', type=float, default=0.5)
    ap.add_argument('--max-attack', type=float, default=0.5)
    ap.add_argument('--bridge', type=float, default=0.3)
    ap.add_argument('--basis', default='dft', choices=['dct', 'dft'])
    ap.add_argument('--folders', help='comma-separated folder names to build (default: all)')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--no-sfizz', action='store_true')
    ap.add_argument('--preview', action='store_true', help='also write the A/B preview per note (large)')
    ap.add_argument('--bits', type=int, default=16, choices=[16, 24], help='FLAC bit depth (default 16: the SSO sources are 16-bit)')
    ap.add_argument('--combine-only', action='store_true',
                    help='only (re)write the per-instrument and per-variant .sfz files from an existing output')
    a = ap.parse_args(argv)
    if a.combine_only:
        for folder, inst in combine(a.out):
            ks = ', '.join(f'{_note_name(k)}={v}' for v, k in inst['keyswitches'].items()) if inst['keyswitches'] else 'no keyswitch'
            print(f'  {folder + ".sfz":28s} {inst["regions"]:3d} regions, {len(inst["variants"])} articulation(s): {ks}')
        return 0

    folders = {f.strip().lower() for f in a.folders.split(',')} if a.folders else None
    inc, exc = find_sustains(folders=folders)
    if a.limit:
        inc = inc[:a.limit]
    print(f'{len(inc)} sustained files in {len({d for _, d, _, _ in inc})} folders; {len(exc)} skipped')
    os.makedirs(a.out, exist_ok=True)
    manifest = [(rel, 'skipped', why) for rel, why in exc]
    groups = defaultdict(list)                      # (folder, variant) -> [(Replication, dynamic)]
    rows, t0, fails = [], time.time(), 0
    for i, (path, folder, variant, dyn) in enumerate(inc, 1):
        rel = os.path.relpath(path, SSO)
        out_dir = os.path.join(a.out, folder)
        try:
            r = replicate_unaltered(path, out_dir, a.loop, method='bridge', basis=a.basis, max_attack_s=a.max_attack,
                                    bridge_s=a.bridge, sfizz=not a.no_sfizz, preview=a.preview, note_sfz=False, bits=a.bits)
            groups[(folder, variant)].append((r, dyn))
            rows.append((rel, r))
            held = 'ok' if (r.sfizz is None or r.sfizz.get('ok')) else 'sfizz hold FAIL'
            manifest.append((rel, 'ok', f'{variant} {dyn or "-"} key {r.keycenter} attack {r.attack_s:.2f}s '
                                        f'tail ncc {r.continuity.get("tail_ncc", float("nan")):.3f} {held}'))
        except Exception as e:
            fails += 1
            manifest.append((rel, 'failed', str(e)[:120]))
        if i % 25 == 0 or i == len(inc):
            dt = time.time() - t0
            print(f'  {i:4d}/{len(inc)}  {dt:5.0f}s  ({dt / i:.2f} s/note)  {fails} failed')

    n_sfz = 0
    for (folder, variant), reps in sorted(groups.items()):
        p = os.path.join(a.out, folder, f'{variant}.sfz')
        write_variant_sfz(p, variant, folder, reps)
        n_sfz += 1
    for folder, inst in combine(a.out):
        n_sfz += 1

    with open(os.path.join(a.out, 'manifest.md'), 'w') as fh:
        fh.write('| file | status | detail |\n|---|---|---|\n')
        for row in manifest:
            fh.write('| ' + ' | '.join(row) + ' |\n')
    with open(os.path.join(a.out, 'summary.md'), 'w') as fh:
        fh.write('| file | key | attack s | loop s | tail ncc | tail resid dB | harm step dB (loop own) '
                 '| band step dB (rec) | untouched | sfizz |\n|' + '---|' * 10 + '\n')
        for rel, r in rows:
            c = r.continuity
            s = 'ok' if (r.sfizz and r.sfizz.get('ok')) else ('FAIL' if r.sfizz else 'n/a')
            fh.write(f'| {rel} | {r.keycenter}{-r.tune_cents:+.0f}c | {r.attack_s:.3f} | {r.loop_seconds:.3f} | '
                     f'{c.get("tail_ncc", float("nan")):.3f} | {c.get("tail_residual_db", float("nan")):.0f} | '
                     f'{c["harm_step_db_wmean"]:.2f} ({c.get("loop_harm_step_db_wmean", 0):.2f}) | '
                     f'{c["band_step_db_mean"]:.1f} ({c["rec_band_step_db_mean"]:.1f}) | '
                     f'{"yes" if r.loop_untouched else "NO"} | {s} |\n')
    print(f'\n{len(rows)} notes replicated, {fails} failed, {n_sfz} .sfz files, {time.time() - t0:.0f}s')
    print('wrote', os.path.join(a.out, 'manifest.md'), 'and', os.path.join(a.out, 'summary.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
