#!/bin/sh
conda env create -f environment.yml
conda activate kim
wait

pip install joblib
pip instal hydroeval
pip install pytest
pip install equinox

# CPU-only (Linux/macOS/Windows)
pip install -U jax
# GPU (NVIDIA, CUDA 12)
# pip install -U "jax[cuda12]"
pip install optax

# pip install parsl
# pip install -U "ray[data,train,tune,serve]"
# pip install torch