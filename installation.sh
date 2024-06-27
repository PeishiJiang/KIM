#!/bin/sh
# conda env create -f environment.yml
# conda activate kim

pip install parsl
pip install joblib
pip install -U "ray[data,train,tune,serve]"
pip install torch
pip instal hydroeval
pip install pytest