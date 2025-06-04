# Use a lightweight Python 3.9 base image
FROM python:3.9-slim

# Install system dependencies for OpenCV and other libraries in one layer
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install PyTorch and torchvision CPU-only versions
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.1 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install NumPy with a version compatible with PyTorch 2.0.0
RUN pip install --no-cache-dir numpy==1.24.4

# Install remaining Python dependencies
RUN pip install --no-cache-dir Pillow opencv-python pandas fastparquet argparse logging multiprocessing
