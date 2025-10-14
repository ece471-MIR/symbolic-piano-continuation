#!/bin/bash

uv run src/train.py \
    --train-config configs/train_config.json \
    --model RWKV \
    --train-data data/filtered/train \
    --val-data data/filtered/val
