#!/usr/bin/env bash
# stolen from https://arxiv.org/abs/2509.12267

if [ $# -ne 3 ]; then
    echo 'usage: $0 input.json out_dir n_samples'
    exit
fi

IN_ABS=$(realpath "$1")
OUT_ABS=$(realpath "$2")
mkdir -p $OUT_ABS
#USER="christianzhouzheng"
IMAGE="christianzhouzheng/rwkv-mirex:latest"
podman pull "$IMAGE"
podman run --rm \
    -v "$IN_ABS:/app/input.json:ro" \
    -v "$OUT_ABS:/app/output" \
    "$IMAGE" "/app/input.json" \
    "/app/output" "$3"
