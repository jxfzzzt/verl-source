#! /bin/bash

# clone verl source code
git clone git@github.com:jxfzzzt/verl-source.git
cd verl-source

# create conda environment

conda create -n verl python=3.10 -y
conda activate verl
pip install -e .

pip install -r requirements.txt
