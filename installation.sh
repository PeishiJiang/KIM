#!/bin/sh
conda env create -f environment.yml
conda activate kim
wait

pip install joblib
pip instal hydroeval
pip install pytest
# pip install parsl
# pip install -U "ray[data,train,tune,serve]"
# pip install torch