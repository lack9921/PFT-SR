# PFT-SR | CVPR 2025
# Progressive Focused Transformer for Single Image Super-Resolution
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 只需装 PyTorch 之外的小包
RUN pip install --no-cache-dir \
    addict fairscale future lmdb numpy opencv-python Pillow \
    pyyaml requests scikit-image scipy tqdm yapf

WORKDIR /workspace
COPY . /workspace

# 编译 CUDA kernel + 安装项目
RUN cd ops_smm && python setup.py build_ext --inplace
RUN python setup.py develop

CMD ["/bin/bash"]
