"""dctjoin.sfz — write the replication as a playable SFZ instrument."""
from __future__ import annotations

import math
import os


def midi_from_hz(f: float) -> float:
    return 69.0 + 12.0 * math.log2(f / 440.0)


def key_and_tune(f0: float) -> tuple[int, float]:
    """(nearest MIDI key, tune in cents to reach f0 exactly)."""
    m = midi_from_hz(f0)
    k = int(round(m))
    return k, 100.0 * (m - k)


def assign_key_ranges(keycenters: list[int], max_stretch: int = 7) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping lokey/hikey ranges split at the midpoints between neighbouring
    notes, each note stretched at most ``max_stretch`` semitones in either direction."""
    order = sorted(range(len(keycenters)), key=lambda i: keycenters[i])
    ks = [keycenters[i] for i in order]
    ranges = [None] * len(ks)
    for j, k in enumerate(ks):
        lo = max(0, k - max_stretch) if j == 0 else max(k - max_stretch, (ks[j - 1] + k) // 2 + 1)
        hi = min(127, k + max_stretch) if j == len(ks) - 1 else min(k + max_stretch, (k + ks[j + 1]) // 2)
        ranges[order[j]] = (int(lo), int(max(lo, hi)))
    return ranges


def region_text(sample: str, keycenter: int, tune_cents: float, loop_start: int, loop_end: int,
                lokey: int | None = None, hikey: int | None = None, release: float = 0.25) -> str:
    lo = keycenter if lokey is None else lokey
    hi = keycenter if hikey is None else hikey
    return (f'<region>\n'
            f'sample={sample}\n'
            f'pitch_keycenter={keycenter} lokey={lo} hikey={hi} tune={int(round(tune_cents))}\n'
            f'loop_mode=loop_continuous loop_start={loop_start} loop_end={loop_end}\n'
            f'ampeg_release={release:.3f}\n')


def write_sfz(path: str, regions: list[dict], header: str = '') -> str:
    """regions: dicts with sample (path relative to the sfz), keycenter, tune_cents, loop_start,
    loop_end, optional lokey/hikey/release."""
    lines = [f'// {header}' if header else '// dctjoin replication', '<control>', f'default_path={os.path.dirname(regions[0]["sample"]) + "/" if os.path.dirname(regions[0]["sample"]) else ""}',
             '<global>', 'amp_veltrack=0']
    for r in regions:
        lines.append(region_text(os.path.basename(r['sample']), r['keycenter'], r['tune_cents'], r['loop_start'],
                                 r['loop_end'], r.get('lokey'), r.get('hikey'), r.get('release', 0.25)))
    text = '\n'.join(lines) + '\n'
    with open(path, 'w') as fh:
        fh.write(text)
    return text
