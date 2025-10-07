#!/bin/bash

if [ ! -f /tmp/data.tar.gz ]; then
    echo "downloading dataset"

    wget \
        -O /tmp/data.tar.gz \
        https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-pruned-ext.tar.gz?download=true
else
    echo "dataset already downloaded"
fi
echo "dataset saved to /tmp/data.tar.gz"

echo 'checking...'
if ! sha256sum --check data.sha; then
    echo 'Download failed!'
fi
