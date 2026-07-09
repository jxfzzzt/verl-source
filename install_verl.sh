#! /bin/bash

# Python 3.10. verl v0.4.1 recommended runtime stack:
# PyTorch 2.6.0 + CUDA 12.4 wheels, vLLM 0.8.5, SGLang 0.4.6.post5.
# clone verl source code
git clone git@github.com:jxfzzzt/verl-source.git
cd verl-source

# create conda environment
conda create -n verl python=3.10 -y
conda activate verl

# install the dependencies
python -m pip install -U pip
python -m pip install --force-reinstall "setuptools<70"
python -m pip install --no-deps -e .
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh