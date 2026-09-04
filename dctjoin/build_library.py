"""Replicate every sustained note of the Sonatina Symphonic Orchestra as recorded attack +
untouched dctloop loop, and write one playable .sfz per instrument variant.

    python3 dctjoin/build_library.py [-o dctjoin_library] [--loop 0.5] [--max-attack 0.5]
                                     [--folders trumpet,horns] [--limit N] [--no-sfizz]
    python3 dctjoin/build_library.py -o dctjoin_library --combine-only     # just the .sfz files
    python3 dctjoin/build_library.py --transcode-from dctjoin_library -o dctjoin_library_ogg --format ogg --quality 1

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
from dctjoin.decay import replicate_decaying, region_lines_decay  # noqa: E402

# pitched decaying families: attack + loop of the flattened body + the decay handed to the SFZ
# envelope (dctjoin.decay); built with --decay
DECAY_PITCHED = {'grand piano', 'harp', 'vibraphone', 'celeste', 'marimba', 'harpsichord', 'crotales', 'percussion'}

SSO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Samples')

DECAY_FAMILIES = {'grand piano', 'marimba', 'crotales', 'vibraphone', 'glockenspiel', 'chimes', 'harp',
                  'xylophone', 'percussion', 'harpsichord', 'celeste'}
# the organ sustains (one subfolder per stop, notes named by MIDI number: 'organ/great-flute4ft/036-C.flac')
NESTED_SUSTAIN = {'organ'}
# inside the percussion folder only these are pitched, decaying and note-named
PERCUSSION_PITCHED = {'glockenspiel', 'xylophone', 'chimes', 'timpani'}
SKIP_TOKENS = {'piz', 'pizz', 'stc', 'stacc', 'stac', 'spic', 'spicc', 'spiccato', 'marc', 'marcato', 'trm',
               'trem', 'tremolo', 'col', 'legno', 'sord', 'mute', 'muted', 'roll', 'crsc', 'mech', 'rr1', 'rr2', 'rr3'}
IGNORE_TOKENS = {'pb', 'loop', 'lh', 'rh'}          # SSO's pre-looped horns 'horns-sus-ff-a#2-PB-loop'; timpani left/right hand
DYNAMICS = ['pp', 'p', 'mp', 'mf', 'f', 'ff']
DYNAMIC_ALIASES = {'soft': 'p', 'hard': 'f'}       # celeste-c4-soft / -hard
_NOTE = re.compile(r'^[a-g](#|b)?-?\d$', re.I)
_MIDI = re.compile(r'^\d{3}$')


def parse_name(stem: str, prefix: str | None = None) -> dict | None:
    """{'variant', 'dynamic', 'note', 'midi', 'skip'} from an SSO file stem, or None if there is no note.
    The note is a note token ('a#4') or a 3-digit MIDI number ('036-C' -> 36, letter ignored).
    ``prefix`` (a subfolder name such as an organ stop) is prepended to the variant."""
    toks = [t for t in re.split(r'[-_ ]', stem) if t]
    note_i = [i for i, t in enumerate(toks) if _NOTE.match(t)]
    midi = None
    if note_i:
        ni = note_i[-1]
        note = toks[ni]
        rest = [t for i, t in enumerate(toks) if i != ni]
    else:
        mi = [i for i, t in enumerate(toks) if _MIDI.match(t) and 0 <= int(t) <= 127]
        if not mi:
            return None
        midi = int(toks[mi[0]])
        note = toks[mi[0]]
        rest = [t for i, t in enumerate(toks) if i != mi[0] and not re.match(r'^[a-g](#|b)?$', t, re.I)]
    rest = [t for t in rest if t.lower() not in IGNORE_TOKENS]
    dyn = [DYNAMIC_ALIASES.get(t.lower(), t.lower()) for t in rest if DYNAMIC_ALIASES.get(t.lower(), t.lower()) in DYNAMICS]
    variant = [t for t in rest if DYNAMIC_ALIASES.get(t.lower(), t.lower()) not in DYNAMICS]
    if prefix:
        variant = [prefix.replace('-', '_')] + variant
    return dict(variant='_'.join(variant).lower().replace(' ', '_') or 'default', dynamic=(dyn[-1] if dyn else None),
                note=note, midi=midi, skip=any(t.lower() in SKIP_TOKENS for t in rest))


def _audio_files(folder: str):
    """(relative subdir or '', file) for the audio files directly in ``folder`` and one level down."""
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isdir(p) and not os.path.islink(p):
            for g in sorted(os.listdir(p)):
                if g.lower().endswith(('.wav', '.flac', '.aif', '.aiff')):
                    yield f, g
        elif f.lower().endswith(('.wav', '.flac', '.aif', '.aiff')):
            yield '', f


def find_sustains(root: str = SSO, folders: set | None = None, decay: bool = False):
    """(included: [(path, folder, variant, dynamic, midi)], excluded: [(relpath, reason)]).  With
    ``decay`` the pitched decaying families are selected instead of the sustaining ones."""
    inc, exc = [], []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full) or os.path.islink(full):
            continue
        if folders and d.lower() not in folders:
            continue
        seen_stems = set()
        for sub, f in _audio_files(full):
            rel = os.path.join(d, sub, f) if sub else os.path.join(d, f)
            stem_key = (sub, os.path.splitext(f)[0].lower())
            if stem_key in seen_stems:                    # the same note as .wav and .flac: keep the first
                exc.append((rel, 'duplicate of another format of the same note'))
                continue
            seen_stems.add(stem_key)
            if sub and d.lower() not in NESTED_SUSTAIN:
                exc.append((rel, 'nested folder not handled'))
                continue
            info = parse_name(os.path.splitext(f)[0], prefix=sub or None)
            if decay:
                if d.lower() == 'percussion':
                    if info is None or not (set(info['variant'].split('_')) & PERCUSSION_PITCHED):
                        exc.append((rel, 'unpitched percussion'))
                        continue
                elif d.lower() not in DECAY_PITCHED:
                    exc.append((rel, 'not a pitched decaying family'))
                    continue
            elif d.lower() in DECAY_FAMILIES:
                exc.append((rel, 'decay family'))
                continue
            if info is None:
                exc.append((rel, 'no note in the name'))
                continue
            if info['skip']:
                exc.append((rel, 'short articulation'))
                continue
            inc.append((os.path.join(full, sub, f) if sub else os.path.join(full, f), d, info['variant'], info['dynamic'], info['midi']))
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
    decaying = any(getattr(r, 'envelope', None) for r, _ in reps)
    lines = [f'// dctjoin replication: {folder} / {variant}, {len(reps)} notes, '
             f'{len(vr)} velocity layer(s).  Recorded attack -> untouched dctloop loop.',
             '<control>', 'default_path=']
    if decaying:
        rel = next(r.release_sfz for r, _ in reps if getattr(r, 'envelope', None))
        lines.append(f'set_hdcc72={rel / 2:.3f}    // release = CC72 * 2 s, as the SSO grand piano; CC64 rings the note on')
    lines += ['<global>', 'loop_mode=loop_continuous']
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
    for v in variants:                                   # the organ: the 8 ft principal is the natural default
        if 'opendiapason8ft' in v:
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
            env = getattr(r, 'envelope', None)
            if env:                                   # decaying note: one region per envelope term, gains add
                lines += region_lines_decay(os.path.basename(r.outputs['audio']), r.keycenter, -r.tune_cents, r.loop_start,
                                            r.loop_end, r.hold_s, env, r.release_sfz, lo, hi, lo_v, hi_v, sw_last)
                n += len(env)
                continue
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
    dec = [r for reps in groups.values() for r, _ in reps if getattr(r, 'envelope', None)]
    if dec:
        lines.append(f'set_hdcc72={dec[0].release_sfz / 2:.3f}    // release = CC72 * 2 s, as the SSO grand piano; CC64 rings the note on')
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


def write_reports(out_dir: str) -> None:
    """manifest.md (every source file: replicated / skipped, with the reason) and summary.md
    (the numbers per replicated note) rebuilt from the .json files in ``out_dir`` plus the
    selection rules, so a partial build (--folders, --decay) never loses the library-wide view."""
    import json
    done = {}
    for jp in [p for f in sorted(os.listdir(out_dir)) if os.path.isdir(os.path.join(out_dir, f))
               for p in glob_json(os.path.join(out_dir, f))]:
        j = json.load(open(jp))
        done[os.path.relpath(j['source'], SSO)] = j
    inc_s, exc_s = find_sustains()
    inc_d, exc_d = find_sustains(decay=True)
    rows = []
    for path, folder, variant, dyn, _midi in inc_s + inc_d:
        rel = os.path.relpath(path, SSO)
        if rel in done:
            j = done[rel]
            c = j.get('continuity', {})
            extra = (f'env fit {j["render_env_err_db"][0]:.1f}/{j["render_env_err_db"][1]:.1f} dB' if j.get('render_env_err_db')
                     else ('hold ok' if (j.get('sfizz') or {}).get('ok') else ''))
            rows.append((rel, 'ok', f'{variant} {dyn or "-"} key {j["keycenter"]} attack {j["attack_s"]:.2f}s '
                                    f'tail ncc {c.get("tail_ncc", float("nan")):.3f} {extra}'))
        else:
            rows.append((rel, 'not built', variant))
    seen = {r[0] for r in rows}
    for rel, why in exc_s:
        if rel not in seen and why != 'decay family':
            rows.append((rel, 'skipped', why)); seen.add(rel)
    for rel, why in exc_d:
        if rel not in seen and why != 'not a pitched decaying family':
            rows.append((rel, 'skipped', why)); seen.add(rel)
    for rel, why in exc_s:
        if rel not in seen:
            rows.append((rel, 'skipped', 'decaying family (unpitched or not built)')); seen.add(rel)
    with open(os.path.join(out_dir, 'manifest.md'), 'w') as fh:
        fh.write('| file | status | detail |\n|---|---|---|\n')
        for row in sorted(rows):
            fh.write('| ' + ' | '.join(row) + ' |\n')
    with open(os.path.join(out_dir, 'summary.md'), 'w') as fh:
        fh.write('| file | key | attack s | loop s | tail ncc | tail resid dB | harm step dB (loop own) '
                 '| band step dB (rec) | untouched | envelope / sfizz |\n|' + '---|' * 10 + '\n')
        for rel in sorted(done):
            j = done[rel]; c = j.get('continuity', {})
            s = (f'env {" + ".join(f"{a:.2f}e^(-t/{t:.2f})" for a, t in j["envelope"])}, render {j["render_env_err_db"][0]:.1f} dB'
                 if j.get('envelope') else ('sfizz ok' if (j.get('sfizz') or {}).get('ok') else 'sfizz n/a'))
            fh.write(f'| {rel} | {j["keycenter"]}{-j["tune_cents"]:+.0f}c | {j["attack_s"]:.3f} | {j["loop_seconds"]:.3f} | '
                     f'{c.get("tail_ncc", float("nan")):.3f} | {c.get("tail_residual_db", float("nan")):.0f} | '
                     f'{c.get("harm_step_db_wmean", float("nan")):.2f} ({c.get("loop_harm_step_db_wmean", 0):.2f}) | '
                     f'{c.get("band_step_db_mean", float("nan")):.1f} ({c.get("rec_band_step_db_mean", float("nan")):.1f}) | '
                     f'{"yes" if j["loop_untouched"] else "NO"} | {s} |\n')


def combine(out_dir: str, remove_note_sfz: bool = True) -> list:
    """Build every <Folder>/<Folder>.sfz (and refresh the per-variant .sfz files) from the
    per-note .json files already in ``out_dir``; optionally delete the per-note .sfz files;
    then rebuild manifest.md and summary.md from the same files."""
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
            src_dir = os.path.basename(os.path.dirname(j['source']))
            prefix = src_dir if src_dir.lower() != folder.lower() else None      # organ stop subfolder
            info = parse_name(os.path.splitext(os.path.basename(j['source']))[0], prefix=prefix)
            if info is None:
                continue
            rep = SimpleNamespace(keycenter=j['keycenter'], tune_cents=j['tune_cents'], loop_start=j['loop_start'],
                                  loop_end=j['loop_end'], release_sfz=j['release_sfz'], outputs=j['outputs'],
                                  envelope=j.get('envelope'), hold_s=j.get('hold_s'))
            variant = info['variant'] if info['variant'] != 'default' else folder.lower().replace(' ', '_')
            groups[variant].append((rep, info['dynamic']))
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
        if len(groups) == 1 and list(groups)[0] == folder.lower().replace(' ', '_'):
            try:                                          # single plain variant: the two files are identical
                os.remove(os.path.join(fdir, f'{list(groups)[0]}.sfz'))
            except OSError:
                pass
        results.append((folder, inst))
    write_reports(out_dir)
    return results


def transcode(src: str, dst: str, fmt: str = 'ogg', quality: float = 1.0) -> int:
    """Convert an existing library (FLAC) into ``dst`` at ``fmt``: every note's audio is
    transcoded, its .json copied with the new audio path, the .sfz files rebuilt, and every
    loop seam of the encoded file measured against the FLAC's (<dst>/seam_check.md)."""
    import json, shutil, time
    import numpy as np
    import soundfile as sf
    from dctloop import seam_metrics
    from dctjoin.unaltered import encode_ogg, pad_after_loop
    t0 = time.time()
    rows, n, worse = [], 0, 0
    for folder in sorted(os.listdir(src)):
        fdir = os.path.join(src, folder)
        if not os.path.isdir(fdir):
            continue
        os.makedirs(os.path.join(dst, folder), exist_ok=True)
        for jp in sorted(glob_json(fdir)):
            j = json.load(open(jp))
            audio = j['outputs']['audio']
            if not os.path.exists(audio):
                audio = os.path.join(os.path.dirname(jp), os.path.basename(audio))
            stem = os.path.splitext(os.path.basename(audio))[0]
            out_audio = os.path.join(dst, folder, f'{stem}.{fmt}')
            x, fs = sf.read(audio, dtype='float64', always_2d=True)
            ls, le = j['loop_start'], j['loop_end']
            if fmt == 'ogg':
                xp = pad_after_loop(x, ls, le, fs)
                tmp = out_audio + '.tmp.wav'
                sf.write(tmp, xp, fs, subtype='PCM_24')
                encode_ogg(tmp, out_audio, quality)
                os.remove(tmp)
            else:
                xp = x
                sf.write(out_audio, x, fs, subtype='PCM_16')
            y, _ = sf.read(out_audio, dtype='float64', always_2d=True)
            ok_len = len(y) == len(xp)
            x = xp
            s_f = seam_metrics(x[ls:le + 1], fs)['seam_flux_ratio']
            s_o = seam_metrics(y[ls:le + 1], fs)['seam_flux_ratio'] if ok_len else float('nan')
            m = min(len(x), len(y))
            snr = 10 * np.log10(np.mean(x[:m] ** 2) / (np.mean((x[:m] - y[:m]) ** 2) + 1e-20))
            flag = (not ok_len) or (s_o > 1.5 * s_f and s_o > 2.0)
            worse += flag
            rows.append((os.path.join(folder, stem), s_f, s_o, snr, os.path.getsize(out_audio) / os.path.getsize(audio), ok_len, flag))
            j['outputs']['audio'] = out_audio
            j['format'] = fmt
            if fmt == 'ogg':
                j['ogg_quality'] = quality
            j['total_seconds'] = len(xp) / fs
            with open(os.path.join(dst, folder, os.path.basename(jp)), 'w') as fh:
                json.dump(j, fh, default=float)
            n += 1
    combine(dst)                                         # .sfz files + manifest.md / summary.md from the .json files
    with open(os.path.join(dst, 'seam_check.md'), 'w') as fh:
        fh.write(f'# loop seam check: {fmt} q{quality:g} vs FLAC\n\n{n} files, {worse} flagged '
                 f'(seam flux ratio > 1.5x the FLAC one and > 2, or a changed sample count)\n\n')
        fh.write('| file | seam flac | seam encoded | SNR dB | size ratio | flagged |\n|---|---|---|---|---|---|\n')
        for name, s_f, s_o, snr, ratio, ok_len, flag in sorted(rows, key=lambda r: -(r[2] / max(r[1], 1e-9))):
            fh.write(f'| {name} | {s_f:.2f} | {s_o:.2f} | {snr:.1f} | {ratio*100:.0f}% | {"YES" if flag else ""}{"" if ok_len else " (length!)"} |\n')
    snrs = np.array([r[3] for r in rows]); ratios = np.array([r[4] for r in rows])
    print(f'{n} files transcoded to {fmt} q{quality:g} in {time.time() - t0:.0f}s: size {np.mean(ratios)*100:.0f}% of FLAC, '
          f'SNR median {np.median(snrs):.1f} dB (min {snrs.min():.1f}), seams flagged {worse}/{n}; wrote {os.path.join(dst, "seam_check.md")}')
    return 0


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
    ap.add_argument('--format', default='flac', choices=['flac', 'wav', 'ogg'], help='audio format (default flac)')
    ap.add_argument('--quality', type=float, default=1.0, help='Vorbis -q:a for --format ogg (default 1: ~105 kbps)')
    ap.add_argument('--decay', action='store_true',
                    help='build the pitched decaying families (grand piano, harp, ...) with dctjoin.decay instead of the sustains')
    ap.add_argument('--hint-octave', type=int, default=0,
                    help='the file names are this many octaves off the standard convention (SSO grand piano: 1); verified per note')
    ap.add_argument('--transcode-from', metavar='DIR',
                    help='do not render: transcode an existing FLAC library DIR into --out at --format/--quality, '
                         'rebuild its .sfz files and check every loop seam against the FLAC')
    ap.add_argument('--combine-only', action='store_true',
                    help='only (re)write the per-instrument and per-variant .sfz files from an existing output')
    a = ap.parse_args(argv)
    if a.transcode_from:
        return transcode(a.transcode_from, a.out, a.format, a.quality)
    if a.combine_only:
        for folder, inst in combine(a.out):
            ks = ', '.join(f'{_note_name(k)}={v}' for v, k in inst['keyswitches'].items()) if inst['keyswitches'] else 'no keyswitch'
            print(f'  {folder + ".sfz":28s} {inst["regions"]:3d} regions, {len(inst["variants"])} articulation(s): {ks}')
        return 0

    folders = {f.strip().lower() for f in a.folders.split(',')} if a.folders else None
    inc, exc = find_sustains(folders=folders, decay=a.decay)
    if a.limit:
        inc = inc[:a.limit]
    print(f'{len(inc)} files in {len({d for _, d, _, _, _ in inc})} folders; {len(exc)} skipped')
    os.makedirs(a.out, exist_ok=True)
    manifest = [(rel, 'skipped', why) for rel, why in exc]
    groups = defaultdict(list)                      # (folder, variant) -> [(Replication, dynamic)]
    rows, t0, fails = [], time.time(), 0
    for i, (path, folder, variant, dyn, midi) in enumerate(inc, 1):
        rel = os.path.relpath(path, SSO)
        out_dir = os.path.join(a.out, folder)
        # a name with a MIDI number (the organ) states its pitch outright; a stem is needed that
        # keeps the stop, since every stop has the same note names
        f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0) if midi is not None else None
        stem = (variant + '-' + os.path.splitext(os.path.basename(path))[0]) if midi is not None else None
        try:
            if a.decay:
                r = replicate_decaying(path, out_dir, a.loop, basis=a.basis, max_attack_s=a.max_attack, sfizz=not a.no_sfizz,
                                       preview=a.preview, bits=a.bits, format=a.format, quality=a.quality,
                                       hint_mult=2.0 ** a.hint_octave, f0=f0, stem=stem)
                held = f'env fit {r.render_env_err_db[0]:.1f}/{r.render_env_err_db[1]:.1f} dB' if r.render_env_err_db else 'no render'
            else:
                r = replicate_unaltered(path, out_dir, a.loop, method='bridge', basis=a.basis, max_attack_s=a.max_attack,
                                        bridge_s=a.bridge, sfizz=not a.no_sfizz, preview=a.preview, note_sfz=False, bits=a.bits,
                                        format=a.format, quality=a.quality, f0=f0, stem=stem)
                held = 'ok' if (r.sfizz is None or r.sfizz.get('ok')) else 'sfizz hold FAIL'
            groups[(folder, variant)].append((r, dyn))
            rows.append((rel, r))
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
    for folder, inst in combine(a.out):          # also rewrites manifest.md / summary.md from all .json files
        n_sfz += 1

    print(f'\n{len(rows)} notes replicated, {fails} failed, {n_sfz} .sfz files, {time.time() - t0:.0f}s')
    print('wrote', os.path.join(a.out, 'manifest.md'), 'and', os.path.join(a.out, 'summary.md'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
