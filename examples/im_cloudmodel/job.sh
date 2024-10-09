#!/bin/bash
#SBATCH -A PIONEERCLOUD
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -J CloudChamberInverseModeling

source /people/jian449/env_kim.sh
# module load python/miniconda24.4.0
# source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
# conda activate kim
# export PYTHONPATH=${PYTHONPATH}:/people/jian449/KIM/src

# echo $PYTHONPATH
# mpirun -n 50 python run_kim.py
# srun -n 50 python run_kim.py
# srun -N 2 python run_kim.py
# srun python run_kim.py
python run_kims.py
wait