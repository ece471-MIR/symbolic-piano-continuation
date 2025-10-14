# symbolic-piano-continuation
Re-implementation of RWKV from MIREX 2025 Symbolic Music Generation Competition. See the dataset[^1], and the reference paper[^2].

Also see the source repository for the model architecture and training routine[^3].

## usage

Note: The Dockerfile and docker_train.sh script were written to be compatible with Kahan (Cooper EE's GPU cluster).
This repository is assumed to be stored  `/zooper2/$USERNAME/mir`. 

### prepping the dataset

See [sym-music-gen/README.md](sym-music-gen/README.md) for data acquisition instructions.

### training

Store this repository in `/zooper2/$USERNAME/mir`.

To create the directory for the model output, run `mkdir sym-music-gen/runs`

Then, create a job by running `sbatch docker_train.sh`

# references

[^1]: [Aria-MIDI Dataset](https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-pruned-ext.tar.gz?download=true)
[^2]: [A TRADITIONAL APPROACH TO SYMBOLIC PIANO CONTINUATION, Zhou-Zheng et al.](https://futuremirex.com/portal/wp-content/uploads/2025/symbolic-music-generation/RWKV.pdf)
[^3]: [christianazinn/mirex2025](https://github.com/christianazinn/mirex2025/tree/john/rwkv-training)
[^3]: [Symusic Document](https://yikai-liao.github.io/symusic/tutorials/midi_operations.html)
[^4]: [RWKV's github](https://github.com/christianazinn/mirex2025/blob/master/sym-music-gen/src/tokenizer.py)
