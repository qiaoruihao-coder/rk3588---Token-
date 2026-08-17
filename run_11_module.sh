#!/bin/bash
# 1.1 Module Launcher - self-contained, no external params needed

# expand /tmp
sudo mount -o remount,size=8G /tmp 2>/dev/null

# build library path
NVIDIA_LIBS=$(find /home/qiaoruihao/sglang_env_py312/lib -type d -name lib 2>/dev/null | tr '\n' ':')
CUDA_LIB=/usr/local/cuda-12.9/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CUDA_LIB}:/usr/lib/x86_64-linux-gnu"

# use GCC 14 for CUDA
export PATH=/usr/local/gcc14-wrap:/usr/local/cuda-12.9/bin:/usr/bin:/bin

# activate venv and run
source /home/qiaoruihao/sglang_env_py312/bin/activate
python3 /home/qiaoruihao/sglang_1_1_module.py
