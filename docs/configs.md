# Configuring KIM

**TBD**
The configurations are written in Python dictionary and passed into the related Python class.

## Configuring data class
Below are the parameters used to configure and initialize the `Data` class:
```python
data_params = {
    "xscaler_type": "minmax",  # scaler for the x (input) data
    "yscaler_type": "minmax",  # scaler for the y (output) data
}
```
- `"xscaler_type"` **(str, default='')**: The type of x data scaler, either `minmax`, `normalize`, `standard`, `log`, or ``
- `"yscaler_type"` **(str, default='')**: The type of y data scaler, either `minmax`, `normalize`, `standard`, `log`, or ``


## Configuring preliminary analysis
Below are the parameters to configure the preliminary analysis and passed into `Data.calculate_sensitivity` method.
```python
sensitivity_params = {
    "method": "pc", "metric": "it-knn",
    "sst": True, "ntest": 100, "alpha": 0.05, "bins":10, "k": 3,
    "n_jobs": 100, "seed_shuffle": 1234,
    "verbose": 1
}
```
- `"method"` **(str, default='gsa')**: The preliminary analysis method, including:
    - `gsa`: the pairwise global sensitivity analysis;
    - `pc`: a modified PC algorithm that include conditional indendpence test for redundancy check/filtering after gsa
- `"metric"` **(str, default='it-bins')**: The metric calculating the sensitivity, including:
    - `it-bins`: the information-theoretic measures (MI and CMI) using the binning approach
    - `it-knn`: the information-theoretic measures (MI and CMI) using the k-nearest-neighbor approach
    - `corr`: the correlation coefficient
- `"sst"` **(bool, default=False)**: Whether to perform the statistical significance test or the shuffle test
- `"ntest"` **(int, default=100)**: The number of shuffled samples in sst
- `"alpha"` **(float, default=0.05)**: The significance level used in the shuffle test
- `"bins"` **(int, default=10)**: The number of bins for each dimension when `"metric"` == "it-bins"
- `"k"` **(int, default=3)**: The number of nearest neighbors when `"metric"` == "it-knn"
- `"n_jobs"` **(int, default=-1)**: The number of processers/threads used by `joblib.Parallel`
- `"seed_shuffle"` **(int, default=1234)**: The random seed number for doing shuffle test
- `"verbose"` **(int, default=0)**: The verbosity level (0: normal, 1: debug)


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