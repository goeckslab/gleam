# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Install NumPy with a version compatible with PyTorch 2.0.0
RUN pip install --no-cache-dir numpy==1.24.4

# Install timm for GigaPath tile encoder (critical for compatibility)
RUN pip install --no-cache-dir timm>=1.0.3

# Install gigapath package for slide encoder
RUN pip install --no-cache-dir git+https://github.com/prov-gigapath/prov-gigapath.git

# Install remaining Python dependencies
RUN pip install --no-cache-dir Pillow opencv-python pandas fastparquet argparse logging multiprocessing

# Install HuggingFace transformers for model loading
RUN pip install --no-cache-dir transformers huggingface-hub

# Set working directory
WORKDIR /workspace

# Create cache directory for HuggingFace models
RUN mkdir -p /workspace/hf_cache
ENV HF_HOME=/workspace/hf_cache
ENV TORCH_HOME=/workspace/hf_cache
