FROM python:3.10-slim

# Install system dependencies, including those required for building matplotlib and headless Chrome for Kaleido
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libgomp1 \
    libopenblas-dev \
    unzip \
    pkg-config \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    chromium \
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
  && rm -rf /var/lib/apt/lists/*

# Pin setuptools <81 to avoid pkg_resources warnings
RUN pip install --no-cache-dir 'setuptools<81.0.0'

# Upgrade pip and wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Install AutoGluon 1.4.0
RUN pip install --no-cache-dir autogluon==1.4.0

# Install pyarrow
RUN pip install --no-cache-dir pyarrow==20.0.0

# Install matplotlib and seaborn for plotting in the script
RUN pip install --no-cache-dir matplotlib seaborn

RUN pip install shap

RUN pip install --upgrade kaleido

RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g yarn
