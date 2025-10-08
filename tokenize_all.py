from __future__ import annotations
from pathlib import Path
from symusic import Score
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
import pickle
import argparse
import json
import random
import concurrent.futures as fut

def create_remi_tokenizer() -> REMI:
    config = TokenizerConfig(
        use_velocities=False,
        encode_ids_splits="no",
        beat_res={(0, 4): 4},
        num_velocities=1,
        use_chords=False,
        use_rests=False,
        use_tempos=True,
        use_time_signatures=False,
        use_programs=False,
        num_tempos=40,
        tempo_range=(40, 250),
    )
    tokenizer = REMI(config)
    return tokenizer

def find_bar_positions(tokens: list) -> list[int]:
    if len(tokens) < 100:
        return []
    
    estimated_bars = len(tokens) // 60
    if estimated_bars < 16:
        return []
    
    # creates estimated bar positions
    bar_interval = len(tokens) // estimated_bars
    return [i * bar_interval for i in range(estimated_bars + 1)]


def extract_16bar_chunks(tokens: list[int], tokenizer: REMI, min_tokens: int = 100) -> list[list[int]]:
    bar_positions = find_bar_positions(tokens)
    
    if len(bar_positions) < 17:  
        return []
    
    # find all valid 16-bar windows
    valid_windows = []
    for start_bar in range(len(bar_positions) - 16):
        start_pos = bar_positions[start_bar]
        end_pos = bar_positions[start_bar + 16]
        chunk = tokens[start_pos:end_pos]
        
        if len(chunk) >= min_tokens:
            valid_windows.append(chunk)
    
    if not valid_windows:
        return []
    
    # randomly select up to 3 chunks from valid windows
    num_chunks = min(3, len(valid_windows))
    return random.sample(valid_windows, num_chunks)

def process_one_file(pkl_path: Path, tokenizer: REMI, out_dir: Path, min_tokens: int):
    try:
        with pkl_path.open("rb") as f:
            track = pickle.load(f)

            # creates a temporary score object to tokenize
            score = Score()
            score.tracks.append(track)
            score.ticks_per_quarter = 960

            # tokenize
            tokens = tokenizer.encode(score)

        if not tokens or not tokens[0]:
            return False, (pkl_path, "No tokens generated")

        track_tokens = tokens[0] # gets the first track's tokens

        token_ids = [t.value if hasattr(t, 'value') else t for t in track_tokens]

        # extracts 16-bar chunks
        chunks = extract_16bar_chunks(token_ids, min_tokens)

        if not chunks:
            return False, (pkl_path, f"No valid chunks (only {len(token_ids)} tokens)")

        base_name = pkl_path.stem
        for i, chunk in enumerate(chunks):
            out_path = out_dir / f"{base_name}_chunk{i}.json"
            with out_path.open("w") as f:
                json.dump({"tokens": chunk, "length": len(chunk)}, f)
        return True, (pkl_path, len(chunks))

    except Exception as e:
            return False, (pkl_path, str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("data"))
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-tokens", type=int, default=100, help="minimum tokens required per chunk")
    ap.add_argument("--limit", type=int, default=None, help="only process this many files per split (for testing)")
    args = ap.parse_args()

    print("Creating REMI tokenizer...")
    tokenizer = create_remi_tokenizer()
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        in_dir = args.base / "quantized" / split
        out_dir = args.base / "tokenized" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        pkl_files = sorted(in_dir.glob("*.pkl"))
        if args.limit:
            pkl_files = pkl_files[:args.limit]

        if not pkl_files:
            print(f"No pickle files in {in_dir}")
            continue

        print(f"\nTokenizing {len(pkl_files)} files from {split}...")
        ok = 0
        total_chunks = 0
        fails = []

        with fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one_file, p, tokenizer, out_dir, args.min_tokens) for p in pkl_files]

            for future in tqdm(fut.as_completed(futures), total=len(pkl_files)):
                success, info = future.result()
                if success:
                    ok += 1
                    total_chunks += info[1]
                else:
                    fails.append(info)

        print(f"[{split}] Done: {ok} files -> {total_chunks} chunks, {len(fails)} failed")

        if fails:
            log = out_dir / "_failed.txt"
            with log.open("w") as f:
                for path, msg in fails:
                    f.write(f"{path}\t{msg}\n")
            print(f"Failure log -> {log}")

if __name__ == "__main__":
    main()   