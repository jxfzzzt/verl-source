#! /bin/bash

# CUDA 12.8
# clone verl source code
git clone git@github.com:jxfzzzt/verl-source.git
cd verl-source

# create conda environment
conda create -n verl python=3.10 -y
conda activate verl

# install the dependencies
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.1
pip install -e .
pip install -r requirements.txt

