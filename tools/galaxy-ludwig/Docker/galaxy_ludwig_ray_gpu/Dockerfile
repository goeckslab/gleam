FROM ludwigai/ludwig-ray-gpu

USER root

# there is an error with the kubernetes-xenial repo
RUN sed -i '/^deb .*kubernetes-xenial/ s/^/#/' /etc/apt/sources.list.d/kubernetes.list || true

RUN apt-get -y update && apt-get install -y unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install -U pip
    
RUN pip install --no-cache-dir 'git+https://github.com/goeckslab/model-unpickler.git' && \
    pip install --no-cache-dir 'git+https://github.com/goeckslab/smart-report.git@17df590f3ceb065add099f37b4874c85bd275014' && \
    pip install --no-cache-dir plotly

RUN useradd -m -s /bin/bash nonrootuser

USER nonrootuser