FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

LABEL maintainer="Elvis" description="PFT-SR Blackwell sm_120 Ready"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common build-essential gcc g++ wget curl git git-lfs \
        vim nano tree zip unzip \
        libgl1 libglib2.0-0 libgl1-mesa-glx liblmdb-dev ca-certificates ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils python3.10-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -q https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && rm get-pip.py && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN pip install --no-cache-dir torch==2.12.0 torchvision && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir ninja packaging

RUN pip install --no-cache-dir \
        addict fairscale future lmdb opencv-python Pillow \
        pyyaml requests scikit-image scipy tqdm yapf \
        tensorboard matplotlib einops timm yacs

WORKDIR /workspace
CMD ["bash"]

