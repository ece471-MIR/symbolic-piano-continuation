#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
from symusic import Score, Track
from tqdm import tqdm 
import concurrent.futures as fut
import argparse
import pickle

TPQ = 960 # ticks per quarter note
GRID_TICKS = TPQ // 4 # because 16th note = 240 ticks

def quantize_track_ticks(track: Track) -> Track:
    t = track.copy()
    t.notes.sort(inplace=True)
    for n in t.notes:
        # snaps start and duration to 1/16 = 0.25 quarters
        n.time = round(n.time / GRID_TICKS) * GRID_TICKS
        n.duration = max(GRID_TICKS, round(n.duration / GRID_TICKS) * GRID_TICKS)
    t.notes.sort(inplace=True)
    return t

def process_one(midi_path: Path, out_dir: Path):
    try:
        # load in tick mode
        score = Score(midi_path)
        # we prefer non-drum tracks; fallback to the first track if all are drums!
        tracks = [tr for tr in score.tracks if not tr.is_drum] or score.tracks[:1]
        ok_count = 0
        for i, tr in enumerate(tracks):
            tq = quantize_track_ticks(tr)
            out_path = out_dir / f"{midi_path.stem}_t{i}_q16.pkl"
            with out_path.open("wb") as f:
                pickle.dump(tq, f)
            ok_count += 1
        return True, (midi_path, ok_count)
    except Exception as e:
        return False, (midi_path, str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("data"))
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8) # if you want to use more CPU cores, works faster
    ap.add_argument("--limit", type=int, default=None,
                help="Only process this many files per split (for quick tests)") # FOR TESTING!! i tested 5 from each split to see if it works
    args = ap.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        in_dir = args.base / "filtered" / split
        out_dir = args.base / "quantized" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        midi_files = sorted(in_dir.rglob("*.mid")) + sorted(in_dir.rglob("*.midi"))
        if args.limit:
            midi_files = midi_files[:args.limit]
        if not midi_files:
            print(f"No MIDI files in {in_dir}")
            continue
        
        print(f"Quantizing {len(midi_files)} files -> {out_dir}")
        ok = 0
        fails = []
        with fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for success, info in tqdm(ex.map(lambda p: process_one(p, out_dir), midi_files), total=len(midi_files)):
                if success:
                    ok += 1
                else:
                    fails.append(info)

        print(f"[{split}] done: {ok} ok, {len(fails)} failed")
        if fails:
            log = out_dir / "_failed.txt"
            with log.open("w") as f:
                for path, msg in fails:
                    f.write(f"{path}\t{msg}\n")
            print(f"Failure log -> {log}")

if __name__ == "__main__":
    main()
