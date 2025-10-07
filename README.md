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

# references

[^1]: [Aria-MIDI Dataset](https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-pruned-ext.tar.gz?download=true)
[^2]: [A TRADITIONAL APPROACH TO SYMBOLIC PIANO CONTINUATION, Zhou-Zheng et al.](https://futuremirex.com/portal/wp-content/uploads/2025/symbolic-music-generation/RWKV.pdf)
