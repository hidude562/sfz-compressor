#!/usr/bin/env python3
"""Build Legato / Sustain SFZ instruments from dctloop recreations of every SSO legato-patch sample.

    python3 build_legato.py -o sfz_out [-q 0.7] [--jobs 6] [--only "Flute Solo 1,Celli"]

Layout: sfz_out/samples/<Instrument dir>/<name>.{wav,flac,sfz,json}   (one recreation per source sample)
        sfz_out/<Patch name> Legato.sfz   (SSO's two-group legato structure over the recreations)
        sfz_out/<Patch name> Sustain.sfz  (the same regions, polyphonic)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfzc.looper import LoopConfig, process_sample  # noqa: E402

SSO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def key_num(v: str) -> int:
    v = v.strip().lower()
    if v.isdigit():
        return int(v)
    m = re.match(r"([a-g])(#|b)?(-?\d)", v)
    return 12 * (int(m.group(3)) + 1) + NOTE[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)


def parse_regions(inc_path: str) -> list[dict]:
    out = []
    for block in re.split(r"<region>", open(inc_path).read())[1:]:
        d = {}
        for line in block.splitlines():
            line = line.split("//")[0]
            for m in re.finditer(r"(\w+)=(.*?)(?=\s+\w+=|\s*$)", line):
                d[m.group(1)] = m.group(2).strip()
        if "sample" in d:
            out.append(d)
    return out


def legato_patches(only: list[str] | None):
    for sfz in sorted(glob.glob(os.path.join(SSO, "* - Performance", "*Legato*.sfz"))):
        name = os.path.splitext(os.path.basename(sfz))[0].replace(" Legato", "")
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        txt = open(sfz).read()
        incs = list(dict.fromkeys(re.findall(r'#include\s+"([^"]+)"', txt)))
        regions = []
        for inc in incs:
            p = os.path.join(os.path.dirname(sfz), inc)
            if os.path.exists(p):
                regions += parse_regions(p)
        seen = set()
        uniq = []
        for d in regions:
            src = d["sample"].replace("\\", "/").replace("../Samples-looped/", "../Samples/")
            src = os.path.normpath(os.path.join(os.path.dirname(sfz), src))
            if not os.path.exists(src):
                alt = os.path.join(SSO, "Samples", os.path.basename(os.path.dirname(src)), os.path.basename(src))
                src = alt if os.path.exists(alt) else src
            key = (src, d.get("lokey"), d.get("hikey"))
            if key in seen or not os.path.exists(src):
                continue
            seen.add(key)
            uniq.append(dict(src=src, lokey=key_num(d.get("lokey", d.get("pitch_keycenter", "60"))),
                             hikey=key_num(d.get("hikey", d.get("pitch_keycenter", "60"))),
                             keycenter=key_num(d.get("pitch_keycenter", "60")), volume=float(d.get("volume", 0) or 0),
                             tune=float(d.get("tune", 0) or 0)))
        if uniq:
            yield name, uniq


def _process(args):
    src, out_dir, q = args
    try:
        r = process_sample(src, out_dir, LoopConfig(q=q, baseline=False, out_format="flac" if src.lower().endswith(".flac") else "wav"))
        return src, r.sfz_path, None
    except Exception as e:  # noqa: BLE001
        return src, None, repr(e)


def sample_regions(sfz_path: str) -> tuple[list[str], dict]:
    """Region blocks of a per-sample SFZ (lines after each <region>) and the join time."""
    txt = open(sfz_path).read()
    blocks = [b.strip("\n") for b in re.split(r"<region>", txt)[1:]]
    m = re.search(r"loop_start=(\d+)", txt)
    return blocks, dict(loop_start=int(m.group(1)) if m else 0)


def write_instruments(name: str, regs: list[dict], built: dict, out_root: str, sample_dir_rel: str) -> None:
    header = ("// Built by sfz_compressor/build_legato.py from dctloop recreations of the SSO samples\n"
              "<control>\nlabel_cc1=Dynamics\nset_cc1=96\n")
    common = ("amp_veltrack=0\ngroup_volume=-29\ngain_cc1=29\nfil_keytrack=80\nfil_keycenter=c4\n")

    def region_text(d: dict, legato: bool) -> str:
        blocks, info = built[d["src"]]
        out = []
        for b in blocks:
            body = re.sub(r"lokey=\d+ hikey=\d+", f"lokey={d['lokey']} hikey={d['hikey']}", b)
            body = re.sub(r"pitch_keycenter=\d+", f"pitch_keycenter={d['keycenter']}", body)
            body = re.sub(r"sample=(\S+)", lambda m: f"sample={sample_dir_rel}/{m.group(1)}", body, count=1)
            vol = d["volume"]
            body = re.sub(r"volume=(-?[\d.]+)", lambda m: f"volume={float(m.group(1)) + vol:.3f}", body)
            if d["tune"]:
                body = body.replace("loop_mode=", f"tune={d['tune']:.0f} loop_mode=", 1)
            has_fil = "fil_type=" in body
            if legato:
                # start right at the loop, fade in, let the decay stages start immediately
                body = re.sub(r"ampeg_hold=[\d.]+", "ampeg_hold=0", body)
                body = body.replace("loop_mode=", f"offset={info['loop_start']} ampeg_attack=0.2 loop_mode=", 1)
            if not has_fil:
                body += "\n  fil_type=lpf_1p cutoff=500 cutoff_cc1=4800"
            out.append("<region>" + body)
        return "\n".join(out)

    legato = [header, f"// {name} - legato (monophonic, first note keeps the recorded attack)\n",
              "<group>\n" + common + "group=1\noff_by=1\noff_mode=time\noff_time=0.5\ntrigger=first\n"]
    legato += [region_text(d, False) for d in regs]
    legato += ["\n<group>\n" + common + "group=1\noff_by=1\noff_mode=time\noff_time=0.5\ntrigger=legato\n"]
    legato += [region_text(d, True) for d in regs]
    with open(os.path.join(out_root, f"{name} Legato.sfz"), "w") as f:
        f.write("\n".join(legato) + "\n")
    sustain = [header, f"// {name} - sustain (polyphonic)\n", "<group>\n" + common]
    sustain += [region_text(d, False) for d in regs]
    with open(os.path.join(out_root, f"{name} Sustain.sfz"), "w") as f:
        f.write("\n".join(sustain) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="sfz_out")
    ap.add_argument("-q", type=float, default=0.7)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--only", default=None, help="comma separated substrings of patch names")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None
    patches = list(legato_patches(only))
    todo = {}
    for name, regs in patches:
        for d in regs:
            inst_dir = os.path.basename(os.path.dirname(d["src"]))
            todo[d["src"]] = os.path.join(a.out, "samples", inst_dir)
    print(f"{len(patches)} patches, {len(todo)} unique samples, {a.jobs} workers", flush=True)
    os.makedirs(a.out, exist_ok=True)
    built = {}
    jobs = [(src, out_dir, a.q) for src, out_dir in todo.items()]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, (src, sfz, err) in enumerate(ex.map(_process, jobs), 1):
            if err:
                print(f"  [{i}/{len(jobs)}] FAILED {os.path.relpath(src, SSO)}: {err}", flush=True)
                continue
            built[src] = sample_regions(sfz)
            if i % 25 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {os.path.relpath(src, SSO)}", flush=True)
    for name, regs in patches:
        regs = [d for d in regs if d["src"] in built]
        if not regs:
            continue
        regs.sort(key=lambda d: d["keycenter"])
        regs[0]["lokey"] = 0          # the outermost samples cover the whole keyboard (pitch-shifted)
        regs[-1]["hikey"] = 127
        inst_dir = os.path.basename(os.path.dirname(regs[0]["src"]))
        write_instruments(name, regs, built, a.out, f"samples/{inst_dir}")
        print(f"wrote {name} Legato/Sustain ({len(regs)} samples)", flush=True)
    json.dump({n: [d["src"] for d in r] for n, r in patches}, open(os.path.join(a.out, "patches.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
