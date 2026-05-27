# PFT-SR | CVPR 2025
# Progressive Focused Transformer for Single Image Super-Resolution
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch 2.5.0 + CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# 依赖
RUN pip install --no-cache-dir \
    addict fairscale future lmdb numpy opencv-python Pillow \
    pyyaml requests scikit-image scipy tqdm yapf

WORKDIR /workspace
COPY . /workspace

# 编译自定义 CUDA kernel (SMM) + 安装项目
RUN cd ops_smm && python setup.py build_ext --inplace
RUN python setup.py develop

CMD ["/bin/bash"]
