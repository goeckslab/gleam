FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies, Python, and Chromium bits needed for Kaleido
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    ca-certificates \
    build-essential \
    gnupg \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    libopenblas-dev \
    unzip \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    lsb-release \
    wget \
    xdg-utils \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Pin setuptools <81 to avoid pkg_resources warnings and upgrade pip/wheel
RUN pip install --no-cache-dir 'setuptools<81.0.0' && \
    pip install --no-cache-dir --upgrade pip wheel

# Install GPU-enabled PyTorch stack before AutoGluon so it picks up CUDA
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2

# Core Python dependencies
RUN pip install --no-cache-dir \
    autogluon==1.4.0 \
    pyarrow==20.0.0 \
    matplotlib \
    seaborn \
    shap

# Update kaleido for plotting support
RUN pip install --no-cache-dir --upgrade kaleido

RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g yarn \
    && rm -rf /var/lib/apt/lists/*
