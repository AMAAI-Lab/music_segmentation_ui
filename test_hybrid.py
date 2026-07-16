#!/usr/bin/env python3
"""Quick test of the tempo-relative Hybrid in segment.py.

Run inside the activated venv, from the project folder:
    source segmentation_env_311/bin/activate
    python test_hybrid.py           # print before/after only
    python test_hybrid.py --write   # also patch results.json so the dashboard shows it

It reuses the Allin1 + Agglomerative boundaries already in results.json (so it
does NOT need Allin1 to run), computes a real BPM per song with librosa, and
compares the OLD fixed-8s hybrid against the NEW 32-beat tempo-relative hybrid.
"""
import argparse
import json
from pathlib import Path

import librosa

from segment import seg_hybrid, _tempo_bpm, SR_TARGET

RESULTS = Path("results.json")
AUDIO = Path("audio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the new Hybrid rows back into results.json")
    args = ap.parse_args()

    data = json.loads(RESULTS.read_text())

    for name, info in data.items():
        al = info["algorithms"]
        if "Allin1 (SOTA)" not in al or "Agglomerative" not in al:
            print(f"\n### {name}: skipped (needs Allin1 + Agglomerative in results.json)")
            continue

        dur = info["duration"]
        y, sr = librosa.load(str(AUDIO / f"{name}.mp3"), sr=SR_TARGET, mono=True)
        bpm = _tempo_bpm(y, sr)
        thr = 32 * 60 / bpm

        old = seg_hybrid(al["Allin1 (SOTA)"], al["Agglomerative"], dur, bpm=None)  # fixed 8s
        new = seg_hybrid(al["Allin1 (SOTA)"], al["Agglomerative"], dur, bpm=bpm)   # 32 beats

        old = [round(float(x), 2) for x in old] if old is not None else []
        new = [round(float(x), 2) for x in new] if new is not None else []

        print(f"\n### {name}   bpm={bpm:.1f}   tiny_thresh: 8.0s -> {thr:.2f}s")
        print(f"  OLD (fixed 8s): {old}")
        print(f"  NEW (32 beats): {new}")
        print(f"  changed: {'YES' if old != new else 'no'}")

        if args.write and new:
            al["Hybrid (Allin1+Agglom)"] = new
            info["tempo"] = round(float(bpm), 2)

    if args.write:
        RESULTS.write_text(json.dumps(data, indent=2))
        print("\n✓ results.json updated — reload the dashboard to see the new Hybrid.")


if __name__ == "__main__":
    main()
