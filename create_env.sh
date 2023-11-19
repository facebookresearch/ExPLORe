#!/bin/bash
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"

conda create -n explore python=3.10
conda activate explore
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
pip install "cython<3" patchelf
