#! /bin/bash

# CUDA 12.8
# clone verl source code
git clone git@github.com:jxfzzzt/verl-source.git
cd verl-source

# create conda environment
conda create -n verl python=3.10 -y
conda activate verl

# install the dependencies
pip install -e .
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install -r requirements.txt