#!/usr/bin/env python3
"""Render a MIDI file through the built SFZ instruments with sfizz (pysfizz), one synth per channel.

    python3 midi_render.py song.mid -o song.wav --map "0=Flute Solo 1 Legato,1=Celli Sustain,..." [--drums]

Channel map entries are "<channel>=<sfz name without .sfz>" (files looked up in --sfz-dir), or a path.
Channel 9 (drums) gets a one-shot kit built from the SSO Percussion samples unless --no-drums.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfzc import dsp  # noqa: E402
from sfzc.render import renderer_gain_db  # noqa: E402

SSO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERC = os.path.join(SSO, "Samples", "Percussion")

# General MIDI drum note -> (sample glob, velocity layers by name suffix)
GM_DRUMS = {
    35: ["bass_drum-f.wav"], 36: ["bass_drum-f.wav"],
    38: ["snare-rh-p.wav", "snare-rh-mf.wav", "snare-rh-ff.wav"], 40: ["snare-lh-p.wav", "snare-lh-mf.wav", "snare-lh-ff.wav"],
    37: ["castanets-rr1.wav"], 39: ["castanets-rr2.wav"],
    42: ["shaker-hi-a.wav"], 44: ["shaker-hi-b.wav"], 46: ["cabasa-a.wav"],
    49: ["piatti.wav"], 57: ["cymbal_roll-b.wav"], 51: ["finger_cymbal-hi.wav"], 53: ["finger_cymbal-lo.wav"],
    41: ["conga-mut-rr1.wav"], 43: ["conga-opn-rr1.wav"], 45: ["conga-opn-rr2.wav"], 47: ["conga-slp-rr1.wav"], 48: ["conga-slp-rr2.wav"], 50: ["conga-opn-rr1.wav"],
    54: ["tambourine-hit-mf.wav"], 56: ["tambourine-hit-f.wav"], 69: ["cabasa-b.wav"], 70: ["shaker-lo-a.wav"], 75: ["castanets-rr1.wav"],
}


def build_drum_kit(path: str) -> str:
    lines = ["// one-shot GM drum kit from SSO Percussion samples", "<group> loop_mode=one_shot amp_veltrack=60 ampeg_release=0.3"]
    for note, files in GM_DRUMS.items():
        files = [f for f in files if os.path.exists(os.path.join(PERC, f))]
        if not files:
            continue
        n = len(files)
        for i, f in enumerate(files):
            lo = 1 + i * 127 // n
            hi = (i + 1) * 127 // n
            rel = os.path.relpath(os.path.join(PERC, f), os.path.dirname(os.path.abspath(path)))
            lines.append(f"<region> sample={rel} key={note} lovel={lo} hivel={hi}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def midi_events(path: str):
    """(events per channel as [(time_s, kind, a, b)], total seconds)."""
    import mido

    m = mido.MidiFile(path)
    per_ch: dict[int, list] = {}
    t = 0.0
    for msg in m:  # merged, with delta times in seconds (tempo map applied)
        t += msg.time
        if msg.type in ("note_on", "note_off"):
            kind = "off" if msg.type == "note_off" or msg.velocity == 0 else "on"
            per_ch.setdefault(msg.channel, []).append((t, kind, msg.note, msg.velocity))
        elif msg.type == "control_change":
            per_ch.setdefault(msg.channel, []).append((t, "cc", msg.control, msg.value))
        elif msg.type == "pitchwheel":
            per_ch.setdefault(msg.channel, []).append((t, "pw", msg.pitch, 0))
    return per_ch, t


def render_channel(sfz: str, events: list, total_s: float, sr: int, block: int = 256, tail_s: float = 3.0) -> np.ndarray:
    import pysfizz

    synth = pysfizz.Synth(sample_rate=sr, block_size=block)
    if not synth.load_sfz_file(os.path.abspath(sfz), quiet=True):
        raise RuntimeError(f"sfizz could not load {sfz}")
    n_total = int((total_s + tail_s) * sr)
    n_blocks = n_total // block + 1
    ev = sorted(events, key=lambda e: e[0])
    ei = 0
    out = np.zeros((n_blocks * block, 2), np.float32)
    for b in range(n_blocks):
        t0 = b * block
        while ei < len(ev) and int(ev[ei][0] * sr) < t0 + block:
            t_s, kind, a, v = ev[ei]
            d = max(0, int(t_s * sr) - t0)
            if kind == "on":
                synth._synth.note_on(d, a, v)
            elif kind == "off":
                synth._synth.note_off(d, a, 0)
            elif kind == "cc":
                synth._synth.cc(d, a, v)
            elif kind == "pw":
                synth._synth.pitch_wheel(d, a)
            ei += 1
        left, right = synth._synth.render_block()
        out[t0: t0 + block, 0] = left
        out[t0: t0 + block, 1] = right
    return out * 10 ** (-renderer_gain_db(sr) / 20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("midi")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--sfz-dir", default="sfz_out")
    ap.add_argument("--map", required=True, help='"ch=SFZ name,ch=SFZ name,..."')
    ap.add_argument("--gain", default="", help='optional per-channel gain dB "ch=dB,..."')
    ap.add_argument("--no-drums", action="store_true")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--stems", default=None, help="also write each channel's render to this directory (and a playlist.m3u)")
    ap.add_argument("--ignore-cc", default="1", help="comma-separated controllers to drop (default: 1, the mod wheel, "
                    "which GM files use for vibrato but the SSO-style instruments treat as dynamics)")
    a = ap.parse_args()
    ignore = {int(c) for c in a.ignore_cc.split(",") if c.strip()}
    per_ch, total = midi_events(a.midi)
    cmap = {}
    for item in a.map.split(","):
        ch, name = item.split("=", 1)
        p = name if os.path.exists(name) else os.path.join(a.sfz_dir, name + ("" if name.endswith(".sfz") else ".sfz"))
        cmap[int(ch)] = p
    gains = {int(k): float(v) for k, v in (it.split("=") for it in a.gain.split(",") if it)}
    if not a.no_drums and 9 in per_ch and 9 not in cmap:
        cmap[9] = build_drum_kit(os.path.join(a.sfz_dir, "GM Drums (SSO Percussion).sfz"))
    mix = None
    stems = []
    if a.stems:
        os.makedirs(a.stems, exist_ok=True)
    for ch, evs in sorted(per_ch.items()):
        if ch not in cmap:
            print(f"channel {ch}: {sum(1 for e in evs if e[1] == 'on')} notes, no instrument mapped -> skipped")
            continue
        if not os.path.exists(cmap[ch]):
            print(f"channel {ch}: {cmap[ch]} not found -> skipped")
            continue
        evs = [e for e in evs if not (e[1] == "cc" and e[2] in ignore)]
        y = render_channel(cmap[ch], evs, total, a.sr)
        g = 10 ** (gains.get(ch, 0.0) / 20)
        peak = float(np.abs(y).max())
        print(f"channel {ch}: {os.path.basename(cmap[ch])}: {sum(1 for e in evs if e[1] == 'on')} notes, peak {20*np.log10(peak+1e-9):.1f} dBFS, gain {gains.get(ch, 0.0):+.1f} dB")
        y = y * g
        if a.stems:
            nm = os.path.splitext(os.path.basename(cmap[ch]))[0].replace(" ", "_")
            sp = os.path.join(a.stems, f"ch{ch + 1:02d}_{nm}.wav")
            ys = y * (0.9 / peak / g) if peak * g > 0.9 else y
            dsp.write_audio(sp, ys, a.sr)
            stems.append(sp)
        mix = y if mix is None else (mix + y if len(mix) == len(y) else mix[: min(len(mix), len(y))] + y[: min(len(mix), len(y))])
    if mix is None:
        print("nothing rendered")
        return 1
    peak = float(np.abs(mix).max())
    if peak > 0.9:
        mix *= 0.9 / peak
    dsp.write_audio(a.out, mix, a.sr)
    print(f"wrote {a.out} ({len(mix)/a.sr:.1f} s, {len(cmap)} channels)")
    if a.stems:
        with open(os.path.join(a.stems, "playlist.m3u"), "w") as f:
            f.write("\n".join(os.path.abspath(p) for p in stems + [a.out]) + "\n")
        print(f"stems: {len(stems)} files + playlist in {a.stems}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
