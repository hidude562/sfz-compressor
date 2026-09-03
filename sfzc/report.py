"""HTML listening-bench report for a demo run (summary.json rows from demo_sso.py)."""
from __future__ import annotations

import base64
import html
import os
import subprocess

FAMILIES = [
    ("Woodwinds", ("flute", "piccolo", "oboe", "clarinet", "bassoon", "cor anglais")),
    ("Brass", ("horn", "trumpet", "trombone", "tuba")),
    ("Percussion & keyboards", ("piano", "harp", "vibraphone", "chime", "harpsichord", "organ", "celeste", "crotale",
                                "drum", "marimba")),
    ("Voices", ("chorus", "choir")),
    ("Strings", ("violin", "viola", "cell", "bass ", "basses")),
]


def family_of(label: str) -> str:
    l = label.lower()
    for fam, keys in FAMILIES:
        if any(k in l for k in keys):
            return fam
    return "Other"


def ogg_data_uri(wav_path: str, quality: int = 3) -> str:
    ogg = os.path.splitext(wav_path)[0] + ".ogg"
    if not os.path.exists(ogg) or os.path.getmtime(ogg) < os.path.getmtime(wav_path):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-c:a", "libvorbis", "-q:a", str(quality), ogg],
                       check=True)
    with open(ogg, "rb") as f:
        return "data:audio/ogg;base64," + base64.b64encode(f.read()).decode()


def _f(v, nd=2, unit=""):
    if v is None:
        return "–"
    return f"{v:.{nd}f}{unit}"


def _score_bar(score, base, label_a="recreation", label_b="crossfade"):
    def bar(v, cls, lab):
        w = 0 if v is None else max(2, round(100 * min(1.0, max(0.0, v))))
        return (f'<div class="bar {cls}"><span class="bar-lab">{lab}</span>'
                f'<span class="bar-track"><span class="bar-fill" style="width:{w}%"></span></span>'
                f'<span class="bar-val">{_f(v, 2)}</span></div>')
    return f'<div class="bars">{bar(score, "ours", label_a)}{bar(base, "base", label_b)}</div>'


def _loop_map(r):
    """Proportional map of what the recreation keeps: attack (original audio) + loop region."""
    orig = max(r["orig_dur"], 1e-3)
    dur = r["dur"]
    if r["klass"] == "oneshot" or not r.get("loop_ms"):
        kept = min(1.0, dur / orig)
        return (f'<div class="lmap"><div class="lmap-orig"></div><div class="lmap-keep" style="width:{kept*100:.1f}%"></div>'
                f'<div class="lmap-lab">one-shot · {dur:.2f} s kept of {orig:.2f} s</div></div>')
    loop_s = r["loop_ms"] / 1000
    start = max(0.0, dur - loop_s)
    return (f'<div class="lmap"><div class="lmap-orig"></div>'
            f'<div class="lmap-keep" style="width:{start/orig*100:.1f}%"></div>'
            f'<div class="lmap-loop" style="left:{start/orig*100:.1f}%;width:{loop_s/orig*100:.1f}%"></div>'
            f'<div class="lmap-lab">attack {start*1000:.0f} ms · loop {r["loop_ms"]:.0f} ms ({r["loop_periods"]:.0f} periods) '
            f'· original {orig:.2f} s</div></div>')


def write_report(rows: list[dict], path: str, q: float, sfz_texts: dict[str, str] | None = None,
                 budget: float | None = None, method: str = "hybrid") -> None:
    sfz_texts = sfz_texts or {}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(family_of(r["label"]), []).append(r)
    order = [f for f, _ in FAMILIES] + ["Other"]

    # ---- summary table
    trs = []
    for fam in order:
        for r in groups.get(fam, []):
            ratio = r["size_kb"] / max(r["orig_kb"], 1e-3)
            loop = "one-shot" if r["klass"] == "oneshot" else f'{r["loop_ms"]:.0f} ms'
            sc = _f(r.get("score"), 2)
            bs = _f(r.get("base_score"), 2)
            trs.append(f'<tr><td>{html.escape(r["label"])}</td><td class="mono">{r["klass"]}</td>'
                       f'<td class="num">{loop}</td><td class="num">{r["size_kb"]:.0f} / {r["orig_kb"]:.0f}</td>'
                       f'<td class="num">{ratio:.2f}×</td><td class="num ours">{sc}</td><td class="num base">{bs}</td></tr>')
    summary = ('<div class="tablewrap"><table class="summary"><thead><tr><th>Instrument</th><th>class</th><th>loop</th>'
               '<th>size kB (ours / original)</th><th>ratio</th><th>score</th><th>crossfade</th></tr></thead>'
               f'<tbody>{"".join(trs)}</tbody></table></div>')

    # ---- benches
    sections = []
    for fam in order:
        items = groups.get(fam)
        if not items:
            continue
        cards = []
        for r in items:
            clips = ""
            for lab, key in (("Original", "orig_wav"), ("Recreation · sfizz render", "rec_wav"),
                             ("Crossfade-loop baseline", "xf_wav"), ("Recreation held 8 s", "long_wav")):
                p = r.get(key)
                if p and os.path.exists(p):
                    clips += (f'<figure class="clip"><figcaption>{lab}</figcaption>'
                              f'<audio controls preload="none" src="{ogg_data_uri(p)}"></audio></figure>')
            t, bt = r.get("terms", {}), r.get("base_terms", {})
            terms = "".join(f'<tr><td>{k}</td><td class="num">{_f(t.get(k), 3)}</td><td class="num">{_f(bt.get(k), 3)}</td></tr>'
                            for k in ("D_spec", "D_loud", "D_seam", "D_var", "D_pitch", "seam_prominence_db", "pumping_rend_db"))
            facts = [f'<b>{r["klass"]}</b>', f'key {r["key"]} ({r["cents"]:+.0f} c)', f'f0 {r["f0"]:.1f} Hz']
            if r.get("B"):
                facts.append(f'B {r["B"]:.1e}')
            if r["klass"] != "oneshot":
                facts += [f'{r["stages"]} envelope stage{"s" if r["stages"] != 1 else ""}',
                          f'residual {"on" if r["residual"] else "off"}', f'{r["K"]} partials']
            if r.get("total_s"):
                facts.append(f'{r["total_s"]:.2f} s of audio on disk')
            if r.get("mode"):
                facts.append(f'path {r["mode"]}')
            if r.get("method") == "auto" and r.get("log"):
                facts.append(str(r["log"][0]))
            dg = r.get("diagnostics") or {}
            if dg and "seam_flux_db" in dg:
                facts.append(f'seam flux {dg["seam_flux_db"]:+.1f} dB vs interior p90'
                             + (f', detune rms {dg["detune_cents_rms"]:.1f} c' if "detune_cents_rms" in dg else ""))
            facts.append(f'{r["size_kb"]:.0f} kB vs {r["orig_kb"]:.0f} kB original')
            sfz = sfz_texts.get(r["label"], "")
            log = "\n".join(r.get("log", []))
            cards.append(f'''
<article class="bench">
  <header>
    <h3>{html.escape(r["label"])}</h3>
    <span class="path mono">{html.escape(r["file"])}</span>
  </header>
  <p class="facts">{" · ".join(facts)}</p>
  {_loop_map(r)}
  {_score_bar(r.get("score"), r.get("base_score")) if r["klass"] != "oneshot" else '<p class="facts">written as a one-shot (no loop needed)</p>'}
  <div class="clips">{clips}</div>
  <details>
    <summary>Metric terms, analysis log and generated SFZ</summary>
    <div class="tablewrap"><table class="terms"><thead><tr><th>term</th><th>recreation</th><th>crossfade</th></tr></thead><tbody>{terms}</tbody></table></div>
    <pre class="log">{html.escape(log)}</pre>
    <pre class="sfz">{html.escape(sfz)}</pre>
  </details>
</article>''')
        sections.append(f'<section class="family"><h2><span class="eyebrow">Family</span>{fam}</h2>{"".join(cards)}</section>')

    n = len(rows)
    n_loop = sum(1 for r in rows if r["klass"] != "oneshot")
    wins = sum(1 for r in rows if r.get("score") is not None and r.get("base_score") is not None and r["score"] >= r["base_score"])
    tot_ours = sum(r["size_kb"] for r in rows)
    tot_orig = sum(r["orig_kb"] for r in rows)
    title = "Sonatina Half-Second Bench" if budget is not None and abs(budget - 0.5) < 1e-6 else (
        f"Sonatina {budget:g}-Second Bench" if budget is not None else "Sonatina Loop Bench")
    if method == "laroche":
        title = "Sonatina Loop-Locked Bench" if budget is None else f"Loop-Locked {title.replace('Sonatina ', '')}"
    budget_line = (f" · hard budget <b>{budget:g} s</b> of audio per sample (all files together)" if budget is not None else "")
    page = f'''<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f3f5f7;--surface:#ffffff;--fg:#1b2230;--muted:#5c6675;--line:#d9dee5;--accent:#0f766e;--accent-soft:#d5efec;--base:#b45309;--base-soft:#f6e3c9;--loop:#0f766e;--keep:#94a3b8;--code:#eef1f4}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0f1418;--surface:#161c22;--fg:#e6eaee;--muted:#98a2ad;--line:#2a323b;--accent:#2dd4bf;--accent-soft:#12403b;--base:#f59e0b;--base-soft:#3d2a0a;--loop:#2dd4bf;--keep:#475569;--code:#0c1114}}}}
:root[data-theme="dark"]{{--bg:#0f1418;--surface:#161c22;--fg:#e6eaee;--muted:#98a2ad;--line:#2a323b;--accent:#2dd4bf;--accent-soft:#12403b;--base:#f59e0b;--base-soft:#3d2a0a;--loop:#2dd4bf;--keep:#475569;--code:#0c1114}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif;margin:0;padding:32px 20px 64px}}
main{{max-width:1040px;margin:0 auto;display:grid;gap:36px}}
h1,h2,h3{{font-family:Fraunces,Georgia,serif;font-weight:500;text-wrap:balance;margin:0}}
h1{{font-size:2.2rem;letter-spacing:-.01em}}
h2{{font-size:1.45rem;display:flex;flex-direction:column;gap:2px;margin-bottom:14px}}
h3{{font-size:1.25rem}}
.eyebrow{{font:500 .72rem/1 "IBM Plex Sans",sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}}
.lede{{max-width:66ch;color:var(--muted);margin:8px 0 0}}
.mono{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.85em}}
.num{{font-variant-numeric:tabular-nums;text-align:right}}
.kpis{{display:flex;flex-wrap:wrap;gap:10px 28px;margin-top:14px;color:var(--muted)}}
.kpis b{{color:var(--fg);font-weight:600}}
.tablewrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th{{text-align:left;font-weight:500;color:var(--muted);border-bottom:1px solid var(--line);padding:6px 10px;white-space:nowrap}}
th:not(:first-child):not(:nth-child(2)){{text-align:right}}
td{{padding:6px 10px;border-bottom:1px solid var(--line)}}
td.ours{{color:var(--accent);font-weight:600}} td.base{{color:var(--base)}}
.family{{display:grid;gap:14px}}
.bench{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px 20px;display:grid;gap:10px}}
.bench header{{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px}}
.path{{color:var(--muted)}}
.facts{{margin:0;color:var(--muted);font-size:.9rem}}
.lmap{{position:relative;height:38px;margin:2px 0}}
.lmap-orig{{position:absolute;inset:0 0 18px 0;background:var(--code);border:1px solid var(--line);border-radius:3px}}
.lmap-keep{{position:absolute;top:0;bottom:18px;left:0;background:var(--keep);opacity:.55;border-radius:3px 0 0 3px}}
.lmap-loop{{position:absolute;top:0;bottom:18px;background:var(--loop);border-radius:2px}}
.lmap-lab{{position:absolute;left:0;bottom:0;font-size:.75rem;color:var(--muted);font-variant-numeric:tabular-nums}}
.bars{{display:grid;gap:4px;max-width:520px}}
.bar{{display:grid;grid-template-columns:86px 1fr 44px;align-items:center;gap:10px;font-size:.8rem}}
.bar-lab{{color:var(--muted)}}
.bar-track{{height:8px;background:var(--code);border-radius:4px;overflow:hidden}}
.bar-fill{{display:block;height:100%;background:var(--accent);border-radius:4px}}
.bar.base .bar-fill{{background:var(--base)}}
.bar-val{{font-variant-numeric:tabular-nums;text-align:right}}
.clips{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px 14px;margin-top:4px}}
.clip{{margin:0;display:grid;gap:4px}}
.clip figcaption{{font-size:.78rem;color:var(--muted)}}
audio{{width:100%;height:36px}}
details{{border-top:1px solid var(--line);padding-top:8px}}
summary{{cursor:pointer;color:var(--accent);font-size:.9rem}}
summary:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
pre{{background:var(--code);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto;font:.78rem/1.45 "IBM Plex Mono",ui-monospace,monospace;margin:10px 0 0}}
pre.log{{color:var(--muted)}}
.method{{max-width:72ch;color:var(--muted);font-size:.92rem;display:grid;gap:8px}}
.method p{{margin:0}}
@media (prefers-reduced-motion: no-preference){{.bar-fill{{transition:width .4s ease}}}}
</style>
<main>
<header>
  <span class="eyebrow">Sonatina Symphonic Orchestra · q = {q}{" · budget " + format(budget, "g") + " s" if budget is not None else ""} · method {method}</span>
  <h1>{title}</h1>
  <p class="lede">Each sample was analysed into per-channel partial tracks plus a noise residual, turned into an
  exactly repeating loop with SFZ envelopes, rendered back with sfizz for the length of the original note, and scored
  against the original. Listen to the pairs; the crossfade loop is the conventional method on the same region.</p>
  <div class="kpis"><span><b>{n}</b> samples</span>{("<span>every sample within <b>" + format(budget, "g") + " s</b> of audio</span>") if budget is not None else ""}<span><b>{n_loop}</b> looped, {n - n_loop} kept as one-shots</span>
  <span>recreation ≥ crossfade in <b>{wins}</b> of {n_loop}</span><span>total size <b>{tot_ours/1024:.1f} MB</b> vs {tot_orig/1024:.1f} MB original</span></div>
</header>
<section>
  <h2><span class="eyebrow">Overview</span>All samples</h2>
  {summary}
</section>
{"".join(sections)}
<section class="method">
  <h2><span class="eyebrow">Reading the numbers</span>How to read this page</h2>
  <p>The score is exp(−Σ wᵢ Dᵢ) over five distortion terms: spectral envelope (dB), BS.1770 loudness envelope (LU),
  seam detectability (transient at the known seam phase plus periodic level pumping), temporal micro-variation
  (a dead loop scores low) and pitch/vibrato mismatch. The weights are engineering defaults, not listening-test
  calibrated; treat scores as a ranking, not an absolute grade. Even a note looped from its own audio scores around
  0.4–0.6 because a natural note keeps evolving where the loop repeats.</p>
  <p>The loop map shows what the recreation keeps of the original note: grey is the recorded attack, teal is the loop.
  Sizes count every sample file the SFZ references (16-bit WAV) against the original sample file.</p>
</section>
</main>
'''
    with open(path, "w") as f:
        f.write(page)
