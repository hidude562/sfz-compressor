"""Command line interface.

    python -m sfzc loop  <sample.wav|dir> -o OUT [-q 0.7] [--stages N] [--no-residual] [--baseline] [--no-verify]
    python -m sfzc score <original.wav> <instrument.sfz> --key 69 [--note-on 4.0]
    python -m sfzc render <instrument.sfz> --key 69 --note-on 3 --dur 4 -o out.wav
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from . import __version__


def _print_metric(m: dict) -> None:
    keys = ["score", "D_spec", "D_loud", "D_seam", "D_var", "D_pitch", "gain_db", "seam_prominence_db", "pumping_rend_db"]
    print("  " + "  ".join(f"{k}={m[k]:.3f}" for k in keys if k in m and isinstance(m[k], float)))


def cmd_loop(args: argparse.Namespace) -> int:
    from .looper import LoopConfig, process_sample

    paths = []
    for p in args.inputs:
        if os.path.isdir(p):
            for ext in ("wav", "flac", "aif", "aiff", "ogg"):
                paths += sorted(glob.glob(os.path.join(p, f"*.{ext}")))
        else:
            paths.append(p)
    if not paths:
        print("no input files", file=sys.stderr)
        return 2
    cfg = LoopConfig(q=args.q, stages=args.stages, residual=(None if args.residual is None else args.residual),
                     K_max=args.partials, n_candidates=args.candidates, verify=not args.no_verify,
                     note_hint=args.note, baseline=args.baseline, out_format=args.format, fileg=not args.no_fileg,
                     hybrid=args.hybrid, f0_method=args.f0_method, max_total_s=args.max_duration,
                     stage_files=args.stage_files, method=args.method, frozen=args.frozen,
                     target_periods=args.target_periods, refine=args.refine, lfo=args.lfo,
                     loop_crossfade_s=args.loop_crossfade, round_robin=args.round_robin,
                     loop_seconds=args.loop_seconds, dct_basis=args.basis, dct_lock=args.lock)
    rows = []
    results = []
    if args.jobs > 1 and len(paths) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(process_sample, p, args.out, cfg): p for p in paths}
            for p in paths:
                fut = next(f for f, pp in futs.items() if pp == p)
                try:
                    results.append((p, fut.result()))
                except Exception as e:
                    print(f"[error] {p}: {e!r}", file=sys.stderr)
    else:
        for p in paths:
            try:
                results.append((p, process_sample(p, args.out, cfg)))
            except Exception as e:  # keep going in batch mode
                print(f"[error] {p}: {e!r}", file=sys.stderr)
                if len(paths) == 1:
                    raise
    for p, r in results:
        print(f"{os.path.basename(p)}: class={r.klass} key={r.key} ({r.cents:+.0f}c) f0={r.f0:.2f}Hz "
              f"loop={r.loop_len / 44100 * 1000:.0f}ms stages={r.stages} residual={r.residual} "
              f"{r.total_audio_s:.2f}s audio of {r.original_duration_s:.2f}s -> {r.size_bytes / 1024:.0f} kB  {r.sfz_path}")
        if r.metric:
            _print_metric(r.metric)
        if r.baseline_metric:
            print("  crossfade baseline:")
            _print_metric(r.baseline_metric)
        if args.verbose:
            print("  " + "\n  ".join(r.info.get("log", [])))
        rows.append(dict(file=p, klass=r.klass, key=r.key, loop_len=r.loop_len, stages=r.stages, size=r.size_bytes,
                         score=(r.metric or {}).get("score"), baseline_score=(r.baseline_metric or {}).get("score")))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from . import dsp
    from .metric import evaluate_recreation
    from .render import render_sfz

    x, sr = dsp.load_audio(args.original)
    seg = dsp.segment_note(x, sr)
    orig = x[seg.onset: seg.end + 1]
    total = len(orig) / sr
    note_on = args.note_on if args.note_on is not None else max(0.05, (seg.release_onset - seg.onset) / sr)
    y = render_sfz(args.sfz, args.key, note_on, total + 0.05, sr=sr)
    held = ((seg.attack_end - seg.onset) / sr, (seg.release_onset - seg.onset) / sr)
    m = evaluate_recreation(orig, y, sr, held_range=held, loop_period_s=args.loop_len, loop_start_s=args.loop_start)
    print(json.dumps({k: v for k, v in m.items()}, indent=1, default=float))
    if args.out:
        dsp.write_audio(args.out, y, sr)
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Seam diagnostics of the first looped region of an SFZ (report section 5)."""
    import re

    from . import dsp
    from .diagnostics import diagnose_loop, format_report

    txt = open(args.sfz).read()
    m = re.search(r"sample=(\S+).*?loop_start=(\d+).*?loop_end=(\d+)", txt, re.S)
    if not m:
        print("no looped region found", file=sys.stderr)
        return 2
    wav = os.path.join(os.path.dirname(os.path.abspath(args.sfz)), m.group(1))
    ls, le = int(m.group(2)), int(m.group(3))
    x, sr = dsp.load_audio(wav)
    locked = orig = None
    jp = os.path.splitext(args.sfz)[0] + ".json"
    if os.path.exists(jp):
        j = json.load(open(jp))
        info = j.get("info", {})
        if "diagnostics" in info and "locked_freqs" in info.get("diagnostics", {}):
            pass
    d = diagnose_loop(x[ls: le + 1], sr, locked, orig)
    print(f"{wav}  loop {ls}..{le}")
    print(format_report(d))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from . import dsp
    from .render import render_sfz

    y = render_sfz(args.sfz, args.key, args.note_on, args.dur, sr=args.sr)
    dsp.write_audio(args.out, y, args.sr)
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sfzc", description=f"sfzc {__version__}: automatic SFZ looping by resynthesis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("loop", help="build a looped SFZ recreation of one or more samples")
    a.add_argument("inputs", nargs="+")
    a.add_argument("-o", "--out", required=True)
    a.add_argument("-q", type=float, default=0.7, help="quality / size knob in [0,1]")
    a.add_argument("--stages", type=int, default=None, help="number of coherent envelope regions (1 or 2)")
    a.add_argument("--residual", dest="residual", action="store_true", default=None)
    a.add_argument("--no-residual", dest="residual", action="store_false")
    a.add_argument("--partials", type=int, default=None, help="max number of partials")
    a.add_argument("--candidates", type=int, default=4, help="loop candidates to render and score")
    a.add_argument("--no-verify", action="store_true", help="skip sfizz rendering + metric")
    a.add_argument("--baseline", action="store_true", help="also build/score a classic crossfade loop")
    a.add_argument("--note", default=None, help="note hint, e.g. a4 (constrains f0 search)")
    a.add_argument("--format", default="wav", choices=["wav", "flac"])
    a.add_argument("--no-fileg", action="store_true")
    a.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="parallel worker processes for directories")
    a.add_argument("--f0", dest="f0_method", default="spectral", choices=["spectral", "pyin"])
    a.add_argument("--max-duration", type=float, default=None,
                   help="hard budget: total seconds of audio over all files written for a sample")
    a.add_argument("--stage-files", default="delay", choices=["delay", "padded"],
                   help="extra stage/noise files: loop-only + delay opcode (sfizz-calibrated) or zero-padded")
    a.add_argument("--method", default="dctloop", choices=["dctloop", "auto", "hybrid", "laroche"],
                   help="dctloop: loop-periodic-grid reconstruction (default); hybrid: tracked partials / "
                        "original-audio hybrid loops; laroche: frozen loop-locked oscillator bank; auto: hybrid vs "
                        "laroche by Metric B")
    a.add_argument("--loop-seconds", type=float, default=None, help="dctloop: target loop length in seconds")
    a.add_argument("--basis", default="dft", choices=["auto", "dct", "dft"], help="dctloop synthesis basis (dft keeps the original phases at the join)")
    a.add_argument("--lock", type=float, default=1.5, help="dctloop harmonic-lock half-width in grid bins")
    a.add_argument("--frozen", default="auto", choices=["auto", "on", "off"], help="laroche: freeze partials")
    a.add_argument("--target-periods", type=float, default=166.0, help="laroche: preferred loop length in periods")
    a.add_argument("--refine", action="store_true", help="laroche: PyTorch MR-STFT refinement of partial/noise gains")
    a.add_argument("--lfo", action="store_true", help="laroche: add gentle pitch/amp LFO opcodes")
    a.add_argument("--loop-crossfade", type=float, default=0.0, help="laroche: loop_crossfade seconds (sfizz/OpenMPT)")
    a.add_argument("--round-robin", type=int, default=1, help="laroche: alternating loop sets")
    a.add_argument("--hybrid", default="auto", choices=["auto", "on", "off"],
                   help="keep original audio in the loop and resynthesise only a phase-closing bridge (sustain class)")
    a.add_argument("-v", "--verbose", action="store_true")
    a.set_defaults(func=cmd_loop)
    s = sub.add_parser("score", help="render an SFZ with sfizz and score it against the original")
    s.add_argument("original")
    s.add_argument("sfz")
    s.add_argument("--key", type=int, required=True)
    s.add_argument("--note-on", type=float, default=None)
    s.add_argument("--loop-len", type=float, default=None, help="loop length in seconds (for the seam term)")
    s.add_argument("--loop-start", type=float, default=None, help="loop start in seconds")
    s.add_argument("-o", "--out", default=None, help="write the rendered audio")
    s.set_defaults(func=cmd_score)
    dg = sub.add_parser("diagnose", help="seam diagnostics (tiled spectral flux, continuity, DC) of an SFZ loop")
    dg.add_argument("sfz")
    dg.set_defaults(func=cmd_diagnose)
    r = sub.add_parser("render", help="render one note of an SFZ with sfizz")
    r.add_argument("sfz")
    r.add_argument("--key", type=int, required=True)
    r.add_argument("--note-on", type=float, default=3.0)
    r.add_argument("--dur", type=float, default=4.0)
    r.add_argument("--sr", type=int, default=44100)
    r.add_argument("-o", "--out", required=True)
    r.set_defaults(func=cmd_render)
    args = ap.parse_args(argv)
    return args.func(args)
