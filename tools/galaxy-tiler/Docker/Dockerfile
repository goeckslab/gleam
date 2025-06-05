FROM debian:stable

LABEL maintainer="Paulo Lyra" \
      version="1.0.0" \
      description="Docker image for PyHIST Galaxy tool"

# Install necessary tools
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update --fix-missing -qq && \
    apt-get install -y -q \
    build-essential \
    libgl1-mesa-glx \
    python3 \
    python3-pip \
    python3-venv \
    openslide-tools \
    python3-openslide \
    libmagic-dev \
    git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /pyhist

# Clone PyHIST repository
RUN git clone https://github.com/manuel-munoz-aguirre/PyHIST.git . && \
    git checkout master

# Create and activate virtual environment, then install Python packages
RUN python3 -m venv /pyhist/venv && \
    /pyhist/venv/bin/pip install --upgrade pip && \
    /pyhist/venv/bin/pip install \
    pandas \
    opencv-python \
    numpy \
    Pillow \
    python-magic \
    openslide-python \
    psutil

# Compile segmentation algorithm
RUN cd src/graph_segmentation/ && \
    make && \
    chmod 755 segment

# Add venv binaries to PATH
ENV PATH="/pyhist/venv/bin:$PATH"

# Make pyhist.py globally executable and accessible
RUN chmod +x /pyhist/pyhist.py && \
    ln -s /pyhist/pyhist.py /usr/local/bin/pyhist

# Ensure src directory is in PYTHONPATH for module imports
ENV PYTHONPATH="/pyhist:$PYTHONPATH"

# Set default command (optional, for debugging)
CMD ["/bin/bash"]
