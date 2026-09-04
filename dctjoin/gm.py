"""dctjoin.gm — a General MIDI soundbank over the replicated library, and a renderer that follows
program changes.

    python3 -m dctjoin.gm build  dctjoin_library_ogg              # -> dctjoin_library_ogg/GM/NNN Name.sfz (+ Drums.sfz, README.md)
    python3 -m dctjoin.gm render song.mid out.wav --bank dctjoin_library_ogg/GM [--play]

Every GM program gets one self-contained .sfz whose ``default_path`` points at the instrument's
folder, so no audio is duplicated.  The SSO is an orchestra: programs with no orchestral
equivalent (guitars, synths, ethnic, FX) are given the nearest timbre and marked as stand-ins
in the README the build writes.  Channel 10 uses a one-shot kit built from the SSO percussion
samples (midi_render.build_drum_kit).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# (GM program name, library folder, variant or None for the folder's default, stand-in?)
GM_PROGRAMS = [
    # 0-7 piano
    ('Acoustic Grand Piano', 'Grand Piano', None, False), ('Bright Acoustic Piano', 'Grand Piano', None, False),
    ('Electric Grand Piano', 'Grand Piano', None, True), ('Honky-tonk Piano', 'Grand Piano', None, True),
    ('Electric Piano 1', 'Vibraphone', None, True), ('Electric Piano 2', 'Vibraphone', None, True),
    ('Harpsichord', 'Grand Piano', None, True), ('Clavinet', 'Grand Piano', None, True),
    # 8-15 chromatic percussion
    ('Celesta', 'Celeste', None, False), ('Glockenspiel', 'Percussion', 'glockenspiel', False),
    ('Music Box', 'Celeste', None, True), ('Vibraphone', 'Vibraphone', None, False),
    ('Marimba', 'Marimba', 'marimba_yarn', False), ('Xylophone', 'Percussion', 'xylophone', False),
    ('Tubular Bells', 'Percussion', 'chimes', False), ('Dulcimer', 'Harp', None, True),
    # 16-23 organ
    ('Drawbar Organ', 'Organ', 'great_opendiapason8ft', True), ('Percussive Organ', 'Organ', 'great_principal4ft', True),
    ('Rock Organ', 'Organ', 'great_opendiapason8ft', True), ('Church Organ', 'Organ', 'great_opendiapason8ft', False),
    ('Reed Organ', 'Organ', 'swell_oboe8ft', True), ('Accordion', 'Organ', 'swell_gamba8ft', True),
    ('Harmonica', 'Organ', 'swell_oboe8ft', True), ('Tango Accordion', 'Organ', 'swell_gamba8ft', True),
    # 24-31 guitar
    ('Acoustic Guitar (nylon)', 'Harp', None, True), ('Acoustic Guitar (steel)', 'Harp', None, True),
    ('Electric Guitar (jazz)', 'Harp', None, True), ('Electric Guitar (clean)', 'Harp', None, True),
    ('Electric Guitar (muted)', 'Marimba', 'marimba_yarn', True), ('Overdriven Guitar', 'Trumpets', None, True),
    ('Distortion Guitar', 'Trombones', None, True), ('Guitar Harmonics', 'Viola', 'viola_hrm', True),
    # 32-39 bass
    ('Acoustic Bass', 'Bass', 'bass_sus', True), ('Electric Bass (finger)', 'Bass', 'bass_sus', True),
    ('Electric Bass (pick)', 'Bass', 'bass_sus', True), ('Fretless Bass', 'Bass', 'bass_sus', True),
    ('Slap Bass 1', 'Basses', None, True), ('Slap Bass 2', 'Basses', None, True),
    ('Synth Bass 1', 'Contrabassoon', None, True), ('Synth Bass 2', 'Bass Trombone', None, True),
    # 40-47 strings
    ('Violin', 'Violin', None, False), ('Viola', 'Viola', 'viola_sus', False), ('Cello', 'Cello', None, False),
    ('Contrabass', 'Bass', 'bass_sus', False), ('Tremolo Strings', '1st Violins', '1st_violins_sus', True),
    ('Pizzicato Strings', '1st Violins', '1st_violins_sus', True), ('Orchestral Harp', 'Harp', None, False),
    ('Timpani', 'Percussion', 'timpani', False),
    # 48-55 ensemble
    ('String Ensemble 1', '1st Violins', '1st_violins_sus', False), ('String Ensemble 2', '2nd Violins', '2nd_violins_sus', False),
    ('Synth Strings 1', 'Violas', 'violas_sus', True), ('Synth Strings 2', 'Celli', 'celli_sus', True),
    ('Choir Aahs', 'Chorus', 'chorus_female', False), ('Voice Oohs', 'Chorus', 'chorus_male', True),
    ('Synth Voice', 'Chorus', 'chorus_female', True), ('Orchestra Hit', 'Trumpets', None, True),
    # 56-63 brass
    ('Trumpet', 'Trumpet', None, False), ('Trombone', 'Tenor Trombone', None, False), ('Tuba', 'Tuba', None, False),
    ('Muted Trumpet', 'Trumpet', None, True), ('French Horn', 'Horn', None, False), ('Brass Section', 'Trumpets', None, False),
    ('Synth Brass 1', 'Horns', None, True), ('Synth Brass 2', 'Trombones', None, True),
    # 64-71 reed
    ('Soprano Sax', 'Clarinet', None, True), ('Alto Sax', 'Clarinet', None, True), ('Tenor Sax', 'Bass Clarinet', None, True),
    ('Baritone Sax', 'Bassoon', None, True), ('Oboe', 'Oboe', None, False), ('English Horn', 'Cor Anglais', None, False),
    ('Bassoon', 'Bassoon', None, False), ('Clarinet', 'Clarinet', None, False),
    # 72-79 pipe
    ('Piccolo', 'Piccolo', None, False), ('Flute', 'Flute', None, False), ('Recorder', 'Flute', None, True),
    ('Pan Flute', 'Alto Flute', None, True), ('Blown Bottle', 'Alto Flute', None, True), ('Shakuhachi', 'Alto Flute', None, True),
    ('Whistle', 'Piccolo', None, True), ('Ocarina', 'Flute', None, True),
    # 80-87 synth lead
    ('Lead 1 (square)', 'Clarinet', None, True), ('Lead 2 (sawtooth)', 'Trumpet', None, True),
    ('Lead 3 (calliope)', 'Flutes', None, True), ('Lead 4 (chiff)', 'Flute', None, True),
    ('Lead 5 (charang)', 'Trumpets', None, True), ('Lead 6 (voice)', 'Chorus', 'chorus_female', True),
    ('Lead 7 (fifths)', 'Trumpets', None, True), ('Lead 8 (bass + lead)', 'Trombones', None, True),
    # 88-95 synth pad
    ('Pad 1 (new age)', 'Violas', 'violas_sus', True), ('Pad 2 (warm)', 'Celli', 'celli_sus', True),
    ('Pad 3 (polysynth)', '2nd Violins', '2nd_violins_sus', True), ('Pad 4 (choir)', 'Chorus', 'chorus_female', True),
    ('Pad 5 (bowed)', 'Violas', 'violas_sus', True), ('Pad 6 (metallic)', 'Vibraphone', None, True),
    ('Pad 7 (halo)', 'Chorus', 'chorus_female', True), ('Pad 8 (sweep)', '1st Violins', '1st_violins_sus', True),
    # 96-103 synth fx
    ('FX 1 (rain)', 'Flutes', None, True), ('FX 2 (soundtrack)', 'Celli', 'celli_sus', True),
    ('FX 3 (crystal)', 'Celeste', None, True), ('FX 4 (atmosphere)', 'Harp', None, True),
    ('FX 5 (brightness)', 'Flutes', None, True), ('FX 6 (goblins)', 'Basses', None, True),
    ('FX 7 (echoes)', 'Vibraphone', None, True), ('FX 8 (sci-fi)', 'Chorus', 'chorus_male', True),
    # 104-111 ethnic
    ('Sitar', 'Harp', None, True), ('Banjo', 'Harp', None, True), ('Shamisen', 'Harp', None, True), ('Koto', 'Harp', None, True),
    ('Kalimba', 'Marimba', 'marimba_yarn', True), ('Bag pipe', 'Oboes', None, True), ('Fiddle', 'Violin', None, True),
    ('Shanai', 'Oboe', None, True),
    # 112-119 percussive
    ('Tinkle Bell', 'Crotales', None, True), ('Agogo', 'Percussion', 'glockenspiel', True),
    ('Steel Drums', 'Vibraphone', None, True), ('Woodblock', 'Marimba', 'marimba_yarn', True),
    ('Taiko Drum', 'Basses', None, True), ('Melodic Tom', 'Marimba', 'marimba_yarn', True),
    ('Synth Drum', 'Marimba', 'marimba_yarn', True), ('Reverse Cymbal', 'Crotales', None, True),
    # 120-127 sound effects
    ('Guitar Fret Noise', 'Harp', None, True), ('Breath Noise', 'Alto Flute', None, True),
    ('Seashore', 'Chorus', 'chorus_female', True), ('Bird Tweet', 'Piccolo', None, True),
    ('Telephone Ring', 'Percussion', 'glockenspiel', True), ('Helicopter', 'Basses', None, True),
    ('Applause', 'Chorus', 'chorus_male', True), ('Gunshot', 'Basses', None, True),
]
assert len(GM_PROGRAMS) == 128


def _groups(lib: str, folder: str) -> dict:
    """{variant: [(rep, dynamic)]} from a library folder's .json files (see build_library.combine)."""
    from dctjoin.build_library import parse_name
    fdir = os.path.join(lib, folder)
    groups = defaultdict(list)
    if not os.path.isdir(fdir):
        return groups
    for f in sorted(os.listdir(fdir)):
        if not f.endswith('.json'):
            continue
        j = json.load(open(os.path.join(fdir, f)))
        src_dir = os.path.basename(os.path.dirname(j['source']))
        info = parse_name(os.path.splitext(os.path.basename(j['source']))[0],
                          prefix=src_dir if src_dir.lower() != folder.lower() else None)
        if info is None:
            continue
        rep = SimpleNamespace(keycenter=j['keycenter'], tune_cents=j['tune_cents'], loop_start=j['loop_start'],
                              loop_end=j['loop_end'], release_sfz=j['release_sfz'], outputs=j['outputs'],
                              envelope=j.get('envelope'), hold_s=j.get('hold_s'))
        variant = info['variant'] if info['variant'] != 'default' else folder.lower().replace(' ', '_')
        groups[variant].append((rep, info['dynamic']))
    return groups


def build_bank(lib: str, out: str | None = None) -> dict:
    """Write <out>/NNN Name.sfz for every GM program (default out = <lib>/GM), Drums.sfz, README.md."""
    from dctjoin.build_library import region_lines, _default_variant
    out = out or os.path.join(lib, 'GM')
    os.makedirs(out, exist_ok=True)
    rows, missing = [], []
    cache = {}
    for prog, (name, folder, variant, standin) in enumerate(GM_PROGRAMS):
        if folder not in cache:
            cache[folder] = _groups(lib, folder)
        groups = cache[folder]
        if not groups:
            missing.append((prog, name, folder)); rows.append((prog, name, folder, variant, 'MISSING', 0)); continue
        v = variant if variant in groups else _default_variant(sorted(groups))
        reps = groups[v]
        decaying = any(getattr(r, 'envelope', None) for r, _ in reps)
        lines = [f'// GM {prog:03d} {name}  <-  SSO {folder} / {v}' + ('  (orchestral stand-in)' if standin else ''),
                 '<control>', f'default_path=../{folder}/']
        if decaying:
            lines.append(f'set_hdcc72={reps[0][0].release_sfz / 2:.3f}')
        lines += ['<global>', 'loop_mode=loop_continuous']
        rl, n = region_lines(reps)
        lines += rl
        p = os.path.join(out, f'{prog:03d} {name}.sfz')
        with open(p, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        rows.append((prog, name, folder, v, 'stand-in' if standin else 'ok', n))
    # drums: the SSO percussion one-shots (midi_render.build_drum_kit)
    try:
        import midi_render
        midi_render.build_drum_kit(os.path.join(out, 'Drums.sfz'))
        drums = 'Drums.sfz: one-shot GM kit from the SSO percussion samples'
    except Exception as e:                                    # pragma: no cover
        drums = f'Drums.sfz not built: {e}'
    with open(os.path.join(out, 'README.md'), 'w') as fh:
        fh.write(f'# GM bank over the dctjoin library `{os.path.basename(os.path.abspath(lib))}`\n\n'
                 'One .sfz per General MIDI program (`NNN Name.sfz`), each pointing at the replicated instrument '
                 'folder next to this one; channel 10 -> `Drums.sfz`.  "stand-in": the SSO has no such instrument, '
                 'the nearest orchestral timbre is used.\n\n' + drums + '\n\n| program | name | SSO folder | variant | status | regions |\n|---|---|---|---|---|---|\n')
        for prog, name, folder, v, status, n in rows:
            fh.write(f'| {prog} | {name} | {folder} | {v} | {status} | {n} |\n')
    return dict(out=out, programs=len(rows) - len(missing), missing=missing, rows=rows)


# ------------------------------------------------------------------------------- rendering

def render_gm(midi: str, out_wav: str, bank: str, sr: int = 44100, play: bool = False, ignore_cc=(1,)) -> str:
    """Render a MIDI file through the bank: each channel's program (program changes split the
    channel into runs), channel 10 -> Drums.sfz.  Uses midi_render.render_channel (sfizz)."""
    import mido
    import numpy as np
    import soundfile as sf
    import midi_render
    m = mido.MidiFile(midi)
    runs = defaultdict(list)                  # (channel, program) -> events
    prog = defaultdict(int)
    t = 0.0
    for msg in m:
        t += msg.time
        ch = getattr(msg, 'channel', None)
        if ch is None:
            continue
        if msg.type == 'program_change':
            prog[ch] = msg.program
        elif msg.type in ('note_on', 'note_off'):
            kind = 'off' if msg.type == 'note_off' or msg.velocity == 0 else 'on'
            runs[(ch, prog[ch])].append((t, kind, msg.note, msg.velocity))
        elif msg.type == 'control_change' and msg.control not in ignore_cc:
            for key in [k for k in runs if k[0] == ch] or [(ch, prog[ch])]:
                runs[key].append((t, 'cc', msg.control, msg.value))
        elif msg.type == 'pitchwheel':
            for key in [k for k in runs if k[0] == ch] or [(ch, prog[ch])]:
                runs[key].append((t, 'pw', msg.pitch, 0))
    total = t
    mix = None
    for (ch, pg), evs in sorted(runs.items()):
        n_on = sum(1 for e in evs if e[1] == 'on')
        if not n_on:
            continue
        if ch == 9:
            sfz = os.path.join(bank, 'Drums.sfz'); label = 'Drums'
        else:
            name = GM_PROGRAMS[pg][0]
            sfz = os.path.join(bank, f'{pg:03d} {name}.sfz'); label = f'{pg:03d} {name}'
        if not os.path.exists(sfz):
            print(f'  channel {ch + 1:2d}: {label}: no .sfz -> skipped'); continue
        y = midi_render.render_channel(sfz, evs, total, sr)
        print(f'  channel {ch + 1:2d}: {label:32s} {n_on:4d} notes, peak {20 * np.log10(np.abs(y).max() + 1e-9):6.1f} dBFS')
        mix = y if mix is None else mix[: min(len(mix), len(y))] + y[: min(len(mix), len(y))]
    if mix is None:
        raise RuntimeError('nothing rendered')
    peak = float(np.abs(mix).max())
    if peak > 0.9:
        mix *= 0.9 / peak
    sf.write(out_wav, mix, sr, subtype='PCM_16')
    print(f'wrote {out_wav} ({len(mix) / sr:.1f} s)')
    if play:
        subprocess.run(['cvlc', '--play-and-exit', out_wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=len(mix) / sr + 15)
    return out_wav


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build'); b.add_argument('lib'); b.add_argument('-o', '--out')
    r = sub.add_parser('render'); r.add_argument('midi'); r.add_argument('out'); r.add_argument('--bank', required=True)
    r.add_argument('--play', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        res = build_bank(a.lib, a.out)
        print(f'{res["programs"]}/128 programs written to {res["out"]}' + (f'; missing folders: {sorted({m[2] for m in res["missing"]})}' if res['missing'] else ''))
        return 0
    render_gm(a.midi, a.out, a.bank, play=a.play)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
