# %% [markdown]
# This notebook is used to use KIM to perform inverse modeling for the cloud chamber model.

# %%
# Libraries
from pathlib import Path
import pandas as pd
import numpy as np

from kim.map import KIM
from kim.data import Data
from kim.mapping_model import MLP

import jax

# %%
jax.devices()

# %% [markdown]
# # Read the data

# %%
# File and folder paths
f_para = Path("./data/PoissonPertb/Output.csv")
f_state = Path("./data/PoissonPertb/Input_np_holodec.csv")


# %%
df_para, df_state = pd.read_csv(f_para),pd.read_csv(f_state)

# %%
y_keys, x_keys = df_para.keys(), df_state.keys()
y, x = df_para.values, df_state.values

# %%
x.shape, y.shape

# %% [markdown]
# # Configurations

# %% [markdown]
# ## Preliminary analysis configuration

# %%
seed_shuffle = 1234
f_data_save = Path("./results/data")


# %%
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


# %% [markdown]
# ## Ensemble learning configuration

# %%
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

f_kim_save1 = Path("./results/map_many2many")
f_kim_save2 = Path("./results/map_many2one")
f_kim_save3 = Path("./results/map_many2one_cond")


# %%
# Mapping parameters for each test below
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
        # 'backend': 'ray',
        'verbose': 1
    },
    'device': None,
}

# %% [markdown]
# Exploratory data analysis
data = Data(x, y, **data_params)
data.calculate_sensitivity(**sensitivity_params)
# Save the sensitivity analysis to disk
data.save(f_data_save)
# data = Data(x, y)
# data.load(f_data_save)

# %%
# Train the inverse mapping
d = jax.numpy.array([0,3,5,7])
d.devices()

# %%
# Initialize three diffferent KIMs
kim1 = KIM(data, map_configs, map_option='many2many')
kim2 = KIM(data, map_configs, mask_option="sensitivity", map_option='many2one')
kim3 = KIM(data, map_configs, mask_option="cond_sensitivity", map_option='many2one')

# Train the mappings
kim1.train()
kim2.train()
kim3.train()


# %%
# Save 
kim1.save(f_kim_save1)
kim2.save(f_kim_save2)
kim3.save(f_kim_save3)

