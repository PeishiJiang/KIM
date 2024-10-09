#!/bin/sh
# conda env create -f environment.yml
# conda activate kim
# wait

# CPU-only (Linux/macOS/Windows)
pip install -U jax
# GPU (NVIDIA, CUDA 12)
#pip install -U "jax[cuda12]"

pip install joblib
pip install hydroeval
pip install pytest
pip install equinox
pip install optax
pip install tqdm
pip install torch torchvision

# pip install parsl
# pip install -U "ray[default]"
# pip install -U "ray[data,train,tune,serve]"