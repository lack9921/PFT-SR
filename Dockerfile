FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir addict fairscale future lmdb opencv-python Pillow pyyaml requests scikit-image scipy tqdm yapf
WORKDIR /workspace
COPY . /workspace
# 入口脚本：首次启动编译 CUDA kernel，后续跳过
RUN printf '#!/bin/bash\nif [ ! -f /workspace/ops_smm/build/lib.linux-x86_64-cpython-310/smm_cuda*.so ]; then\n  echo \"[PFT-SR] First run: compiling SMM CUDA kernel...\"\n  cd /workspace/ops_smm && python setup.py build_ext --inplace\n  cd /workspace && python setup.py develop\nfi\nexec \"\$@\"\n' > /start.sh && chmod +x /start.sh
ENTRYPOINT ["/bin/bash", "/start.sh"]
CMD ["/bin/bash"]
