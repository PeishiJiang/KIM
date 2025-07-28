# Configuring KIM

**TBD**

## Configuring data class
```python
data_params = {
    "xscaler_type": "minmax",  # scaler for the x (input) data
    "yscaler_type": "minmax",  # scaler for the y (output) data
}
```
- `"xscaler_type"`:
- `"yscaler_type"`:


## Configuring preliminary analysis
```python
sensitivity_params = {
    "method": "pc", "metric": "it-knn",
    "sst": True, "ntest": 100, "alpha": 0.05, "k": 3,
    "n_jobs": 100, "seed_shuffle": 1234,
    "verbose": 1
}
```
- `"method"`:
- `"metric"`:
- `"sst"`:
- `"ntest"`:
- `"alpha"`:
- `"n_jobs"`:
- `"seed_shuffle"`:
- `"verbose"`:


## Configuring ensemble learning
```python
map_configs = {
    "model_type": MLP,
    'n_model': 100,
    'ensemble_type': 'ens_random',
    'model_hp_choices': {
        "depth": [1,3,5,6],
        "width_size": [3,6,10]
    },
    'model_hp_fixed': {
        "hidden_activation": 'sigmoid',
        "final_activation": 'leaky_relu',
        "model_seed": 100
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
        'dl_seed': 10,
        'num_train_sample': 400,
        'num_val_sample': 50,
        'batch_size': 64
    },
    'ens_seed': 1024,
    'training_parallel': True,
    'parallel_config': {
        'n_jobs': 2, 
        'backend': 'loky',
        'verbose': 1
    },
    'device': None,
}
```
- `"model_type"`:
- `"n_model"`:
- `"ensemble_type"`:
- `"ens_seed"`:
- `"training_parallel"`:
- `"device"`:
- `"parallel_config"`:
- `"model_hp_choices"`:
- `"model_hp_fixed"`:
- `"optax_hp_choices"`:
- `"optax_hp_fixed"`:
- `"dl_hp_choices"`:
- `"dl_hp_fixed"`: