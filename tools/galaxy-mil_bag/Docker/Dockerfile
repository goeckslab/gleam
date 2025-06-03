# Base image with PyTorch + CUDA 11.7 + cuDNN 8
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

USER root

# Install system libs needed by fastparquet
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libsnappy-dev \
      liblz4-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      pandas \
      fastparquet

RUN useradd -m -s /bin/bash nonrootuser

USER nonrootuser
