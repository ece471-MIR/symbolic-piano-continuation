# symbolic-piano-continuation
Re-implementation of RWKV from MIREX 2025 Symbolic Music Generation Competition. See the dataset[^1], and the reference paper[^2].

## prerequisites

Install
[uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).

Run `uv sync`

## usage

### prepping the dataset

Make sure you downloaded it first: run `./get_data.sh`

Once you have the data, filter into a train/test split: `uv run prep.py`

Then quantize to 16th notes: `uv run quantize_all.py`

- This will create quantized tracks in `data/quantized/train/` and `data/quantized/val/`
- You have options!

    - `--workers N` - number of parallel workers (default: 8)
    - `--limit N` - only process N files per split (useful for testing, highly recommend)
    - `--split train|val|both` - Process specific split (default: both)
    - example: uv run quantize_all.py --limit 5 workers 16

Then tokenize with REMI: `uv run tokenize_all.py`

- This will create training samples by extracting the 16-bar chunks with at least 100 tokens

- Output goes to `data/tokenized/train/` and `data/tokenized/val/`

- More options!
    
    - `--workers N` - number of parallel workers (default: 8)
    - `--limit N` - only process N files per split (for testing)
    - `--split train|val|both` - process specific split (default: both)
    - example: uv run tokenize_all.py --limit 5

# references

[^1]: [Aria-MIDI Dataset](https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-pruned-ext.tar.gz?download=true)
[^2]: [A TRADITIONAL APPROACH TO SYMBOLIC PIANO CONTINUATION, Zhou-Zheng et al.](https://futuremirex.com/portal/wp-content/uploads/2025/symbolic-music-generation/RWKV.pdf)
[^3]: [Symusic Document](https://yikai-liao.github.io/symusic/tutorials/midi_operations.html)
