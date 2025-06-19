"""
This script is used to perform multiple KIMs, based on
different selections of input variables.

"""

from pathlib import Path
import itertools
from functools import reduce
import pandas as pd
import numpy as np
from kim.map import KIM
from kim.data import Data
from kim.mapping_model import MLP

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')


###########################################
# Parameters
###########################################
# Training data
dir_data = Path("./data")
f_para = Path("./data/PoissonPertb/Output.csv")
f_state = Path("./data/PoissonPertb/Input_np_holodec.csv")

# Saving folder
dir_results = Path("./results-holodec")
# dir_data_save = Path("./results/data")

# Training configurations
mask_option = "cond_sensitivity"
map_option = "many2one"
seed_shuffle = 1234
Ns_train = 400
Ns_val = 50
hidden_activation = 'sigmoid'
final_activation = 'leaky_relu'
seed_ens = 1024
seed_predict = 3636
seed_dl = 10
seed_model = 100
training_verbose = 1
n_models = 100

# Points of interests
pts_sets = [['s2'], ['s6'], ['s10'], ['s20'], ['s24']]

# # Locations of the points
# middle_pts = ['s1', 's10', 's19']
# cold_pts = ['s2', 's3', 's9', 's11', 's12', 's18', 's20', 's21', 's27']
# warm_pts = ['s5', 's6', 's7', 's14', 's15', 's16', 's23', 's24', 's25']

# # Multiple criteria
# middle_pts_r = [True, False]
# all_ss_varns = [True, False]
# wstd = [True, False]
# wall_types = ['cold', 'warm', 'both']
# temperature = [True, False]


###########################################
# Sensitivity configurations and
# mapping configurations
###########################################
# Data configuration
data_params = {
    "xscaler_type": "minmax",
    "yscaler_type": "minmax",
}

# Sensitivity analysis configuration
sensitivity_params = {
    "method": "pc", "metric": "it-knn",
    "sst": True, "ntest": 100, "alpha": 0.05, "k": 3,
    "n_jobs": 100, "seed_shuffle": seed_shuffle,
    "verbose": 1
}

# Mapping configurations
map_configs = {
    "model_type": MLP,
    'n_model': n_models,
    'ensemble_type': 'ens_random',
    'model_hp_choices': {
        "depth": [1,3,5,6],
        "width_size": [3,6,10]
    },
    'model_hp_fixed': {
        "hidden_activation": hidden_activation,
        "final_activation": final_activation,
        "model_seed": seed_model
    },
    'optax_hp_choices': {
        'learning_rate': [0.01, 0.005, 0.003],
    },
    'optax_hp_fixed': {
        'nsteps': 300,
        'optimizer_type': 'adam',
    },
    'dl_hp_choices': {
    },
    'dl_hp_fixed': {
        'dl_seed': seed_dl,
        'num_train_sample': Ns_train,
        'num_val_sample': Ns_val,
        'batch_size': 64
    },
    'ens_seed': seed_ens,
    'training_parallel': True,
    'parallel_config': {
        'n_jobs': 20, 
        'backend': 'loky',
        'verbose': 1
    },
    'device': None,
}


###########################################
# Load the training data
###########################################
df_para, df_state = pd.read_csv(f_para),pd.read_csv(f_state)
y, x = df_para.values, df_state.values

y_keys, x_keys = df_para.keys(), df_state.keys()
locs = [k.split('_')[0] for k in x_keys]
times = [k.split('_')[1] for k in x_keys]
varns = [k.split('_')[2] for k in x_keys]
nxkeys = len(x_keys)

Rvarns = ['Rmean', 'Rstd', 'Rskew', 'Rkurt']

###########################################
# Train different mappings for different
# subsets of the input data
###########################################
# kim3 = KIM(data, map_configs, mask_option=mask_option, map_option=map_option)
# combinations = itertools.product(middle_pts_r, all_ss_varns, wstd, wall_types, temperature)
for pts in pts_sets:
    # Get the remove sets
    removed_set = [(locs[i] not in pts) and (varns[i] in Rvarns) for i,_ in enumerate(x_keys)]
    removed_set = np.array(removed_set)
    other_mask = ~removed_set
    label = "-".join(pts) + "R"
    logging.info(f'Combination: {label}; total number of keys: {other_mask.sum()}')

    # Subset the data
    df_state_subset = df_state.iloc[:,other_mask].copy()
    # print(df_state_subset.keys())

    # print(label)
    # print(df_state_subset.keys())
    # print("")
    # Perform sensitivity analysis
    x_subset = df_state_subset.values
    data = Data(x_subset, y, **data_params)
    data.calculate_sensitivity(**sensitivity_params)

    # Train the inverse mappings
    kim = KIM(data, map_configs, mask_option=mask_option, map_option=map_option)
    kim.train()

    # Save data and model
    f_state_subset = dir_data / f'Input-{label}.csv'
    f_kim_save = dir_results / f'KIM-{label}'
    dir_sensitivity_save = dir_results / f'Data-{label}'
    df_state_subset.to_csv(f_state_subset)
    data.save(dir_sensitivity_save)
    kim.save(f_kim_save)