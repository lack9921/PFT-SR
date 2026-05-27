# PFT-SR | CVPR 2025
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 安装依赖（除 PyTorch 外的小包）
RUN pip install --no-cache-dir \
    addict fairscale future lmdb opencv-python Pillow \
    pyyaml requests scikit-image scipy tqdm yapf

# 首次运行脚本：编译 CUDA kernel + 安装项目
RUN printf '#!/bin/bash\n\
echo \"[PFT-SR] Compiling SMM CUDA kernel...\"\n\
cd /workspace/ops_smm && python setup.py build_ext --inplace\n\
cd /workspace && python setup.py develop\n\
exec \"\$@\"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR /workspace
COPY . /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
