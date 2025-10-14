FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# install compiler for numpy-2.3.3
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        build-essential \
        cmake \
        ninja-build \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# file setup
COPY  sym-music-gen /sym-music-gen/
COPY  rwkv /rwkv/
WORKDIR /sym-music-gen

# dependencies
#RUN pip install --no-cache-dir --no-binary symusic -r requirements.txt
#RUN uv pip sync requirements.txt
RUN pip install uv
RUN uv cache clean
RUN uv sync --refresh

# set CUDA_HOME env var
ENV CUDA_HOME /usr/local/cuda-12.1
ENV PATH $CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH $CUDA_HOME/lib64:$LD_LIBRARY_PATH

# training script
CMD ./spc_train.sh
