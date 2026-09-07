#!/usr/bin/env python3
"""Demo: recreate a variety of Sonatina Symphonic Orchestra samples, render them with sfizz,
score them with Metric B, optionally play original vs. recreation, and write an HTML report.

    python3 demo_sso.py -o demo_out [-q 0.7] [--play] [--no-report]
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfzc import dsp  # noqa: E402
from sfzc.looper import LoopConfig, SampleLooper  # noqa: E402
from sfzc.render import render_sfz  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Samples")
DEFAULT_SET = [
    ("Flute (solo, vibrato)", "Flute/flute-a4.wav"),
    ("Oboe (solo)", "Oboe/oboe-a#4.wav"),
    ("Clarinet (solo)", "Clarinet/clarinet-d4.wav"),
    ("Bassoon (solo)", "Bassoon/bassoon-e3.wav"),
    ("Horn (solo)", "Horn/horn-e3.wav"),
    ("Trumpet (solo, swell)", "Trumpet/trumpet-c#5.wav"),
    ("Trombones (section)", "Trombones/trombones-sus-e3.wav"),
    ("Tuba (low)", "Tuba/tuba-sus-e2.wav"),
    ("Violin (solo, vibrato)", "Violin/violin-a#4.wav"),
    ("Celli (section)", "Celli/celli-sus-c3.wav"),
    ("Chorus (female)", "Chorus/chorus-female-a4.wav"),
    ("Grand piano (decaying, inharmonic)", "Grand Piano/FF A3.flac"),
    ("Harp (decaying)", "Harp/harp-c4.wav"),
    ("Vibraphone (decaying)", "Vibraphone/vibraphone-c4.flac"),
    ("Contrabassoon (very low)", "Contrabassoon/contrabassoon-e1.wav"),
    ("Piccolo (very high)", "Piccolo/piccolo-c6.wav"),
    ("Harpsichord (decaying, plucked)", "Harpsichord/Sustains/Low/*_C3_rr1.flac"),
    ("Organ flute 4' (easy, static)", "Organ/great-flute4ft/060-C.flac"),
    ("Bass drum (one-shot)", "Percussion/bass_drum-f.wav"),
]


def resolve(pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(ROOT, pattern)))
    return hits[0] if hits else None


def to_ogg_b64(wav_path: str) -> str:
    ogg = wav_path[:-4] + ".ogg"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-c:a", "libvorbis", "-q:a", "4", ogg],
                   check=True)
    with open(ogg, "rb") as f:
        return "data:audio/ogg;base64," + base64.b64encode(f.read()).decode()


def play(path: str) -> None:
    for cmd in (["paplay", path], ["pw-play", path], ["aplay", "-q", path]):
        try:
            subprocess.run(cmd, check=True, timeout=60)
            return
        except Exception:
            continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="demo_out")
    ap.add_argument("-q", type=float, default=0.7)
    ap.add_argument("--play", action="store_true", help="play original then recreation through the speakers")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--only", default=None, help="comma separated substrings to select demo entries")
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--max-duration", type=float, default=None, help="hard budget in seconds of audio per sample")
    ap.add_argument("--method", default="dctloop", choices=["dctloop", "auto", "hybrid", "laroche"])
    ap.add_argument("--loop-seconds", type=float, default=None, help="dctloop: target loop length in seconds")
    ap.add_argument("--basis", default="dft", choices=["auto", "dct", "dft"], help="dctloop synthesis basis (dft keeps the original phases at the join)")
    ap.add_argument("--no-tail-morph", action="store_true", help="dctloop: do not EQ-morph the recording towards the loop before the join")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--lfo", action="store_true")
    ap.add_argument("--loop-crossfade", type=float, default=0.0)
    ap.add_argument("--play-only", action="store_true", help="play the A/B files of an existing run and exit")
    ap.add_argument("--report-only", action="store_true", help="rebuild report.html from an existing run and exit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.report_only:
        rows = json.load(open(os.path.join(args.out, "summary.json")))
        for r in rows:
            sfz_p = os.path.join(os.path.dirname(r["rec_wav"]), os.path.basename(os.path.dirname(r["rec_wav"])) + ".sfz")
            r["sfz"] = open(sfz_p).read() if os.path.exists(sfz_p) else ""
        write_report(rows, os.path.join(args.out, "report.html"), args.q, args.max_duration, args.method)
        return 0
    if args.play_only:
        rows = json.load(open(os.path.join(args.out, "summary.json")))
        for r in rows:
            if args.only and not any(s.lower() in (r["label"] + r["file"]).lower() for s in args.only.split(",")):
                continue
            print(f"{r['label']}: original ...", flush=True)
            play(r["orig_wav"])
            time.sleep(0.4)
            print(f"{r['label']}: recreation (score {r.get('score') or 0:.2f}) ...", flush=True)
            play(r["rec_wav"])
            time.sleep(0.7)
        return 0
    rows = []
    for label, pat in DEFAULT_SET:
        if args.only and not any(s.lower() in (label + pat).lower() for s in args.only.split(",")):
            continue
        path = resolve(pat)
        if path is None:
            print(f"[skip] {label}: no file for {pat}")
            continue
        t0 = time.time()
        fmt = "flac" if path.lower().endswith(".flac") else "wav"
        cfg = LoopConfig(q=args.q, n_candidates=args.candidates, baseline=True, out_format=fmt,
                         max_total_s=args.max_duration, method=args.method, refine=args.refine, lfo=args.lfo,
                         loop_crossfade_s=args.loop_crossfade, loop_seconds=args.loop_seconds, dct_basis=args.basis,
                         tail_morph=not args.no_tail_morph)
        from sfzc.looper import process_sample

        lp = SampleLooper(path, cfg)  # for name / render protocol only
        outdir = os.path.join(args.out, lp.name)
        r = process_sample(path, outdir, cfg)
        if cfg.method == "dctloop":
            from sfzc.dctloop_backend import DctLoopLooper

            lp = DctLoopLooper(path, cfg)
            lp.analyse_light()
        else:
            lp.analyse()
        note_on, render_s, orig, held = lp.render_protocol()
        orig_p = os.path.join(outdir, "A_original.wav")
        rec_p = os.path.join(outdir, "B_recreation.wav")
        dsp.write_audio(orig_p, orig, lp.sr)
        y = render_sfz(r.sfz_path, r.key, note_on, render_s, sr=lp.sr)
        dsp.write_audio(rec_p, y, lp.sr)
        xf_p = None
        if r.info.get("baseline"):
            yb = render_sfz(r.info["baseline"], r.key, note_on, render_s, sr=lp.sr)
            xf_p = os.path.join(outdir, "C_crossfade_baseline.wav")
            dsp.write_audio(xf_p, yb, lp.sr)
        # a longer held note (10 s) to demonstrate the loop sustaining beyond the original
        long_p = os.path.join(outdir, "D_recreation_held_8s.wav")
        if r.klass != "oneshot":
            yl = render_sfz(r.sfz_path, r.key, 8.0, 9.5, sr=lp.sr)
            dsp.write_audio(long_p, yl, lp.sr)
        else:
            long_p = None
        m = r.metric or {}
        b = r.baseline_metric or {}
        row = dict(label=label, file=os.path.relpath(path, ROOT), klass=r.klass, key=r.key, cents=r.cents, f0=r.f0,
                   B=r.B, loop_ms=r.loop_len / lp.sr * 1000, loop_periods=r.loop_len * r.f0 / lp.sr if r.f0 else 0,
                   stages=r.stages, residual=r.residual, K=r.K, dur=r.duration_s, orig_dur=r.original_duration_s,
                   total_s=r.total_audio_s, budget=args.max_duration, method=args.method, mode=r.info.get("mode"),
                   diagnostics=r.info.get("diagnostics"),
                   size_kb=r.size_bytes / 1024, orig_kb=os.path.getsize(path) / 1024,
                   score=m.get("score"), base_score=b.get("score"),
                   terms={k: m.get(k) for k in ("D_spec", "D_loud", "D_seam", "D_var", "D_pitch", "seam_prominence_db",
                                                "pumping_rend_db", "pumping_orig_db", "gain_db")},
                   base_terms={k: b.get(k) for k in ("D_spec", "D_loud", "D_seam", "D_var", "D_pitch", "seam_prominence_db")},
                   sfz=open(r.sfz_path).read(), orig_wav=orig_p, rec_wav=rec_p, xf_wav=xf_p, long_wav=long_p,
                   log=r.info.get("log", []), secs=time.time() - t0)
        rows.append(row)
        print(f"{label:36s} class={r.klass:8s} loop={row['loop_ms']:7.1f}ms stages={r.stages} "
              f"score={m.get('score', float('nan')):.3f} (crossfade {b.get('score', float('nan')):.3f}) "
              f"audio {r.total_audio_s:.2f}s size {row['size_kb']:.0f}kB vs {row['orig_kb']:.0f}kB  [{row['secs']:.0f}s]", flush=True)
        if args.play:
            print("   playing original ...", flush=True)
            play(orig_p)
            time.sleep(0.4)
            print("   playing recreation ...", flush=True)
            play(rec_p)
            time.sleep(0.6)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "sfz"} for r in rows], f, indent=1, default=float)
    if not args.no_report:
        write_report(rows, os.path.join(args.out, "report.html"), args.q, args.max_duration, args.method)
    return 0


def write_report(rows: list[dict], path: str, q: float, budget: float | None = None, method: str = "hybrid") -> None:
    from sfzc.report import write_report as _wr

    _wr([{k: v for k, v in r.items() if k != "sfz"} for r in rows], path, q, {r["label"]: r["sfz"] for r in rows},
        budget=budget, method=method)
    print(f"report: {path}")


if __name__ == "__main__":
    sys.exit(main())
