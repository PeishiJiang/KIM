# Configuring KIM

The configurations are written in Python dictionary and passed into the related Python class.

## Configuring data class
Below are the parameters used to configure and initialize the `Data` class:
```python
data_params = {
    "xscaler_type": "minmax",  # scaler for the x (input) data
    "yscaler_type": "minmax",  # scaler for the y (output) data
}
```
- `xscaler_type` **(str, default='')**: The type of x data scaler, either `minmax`, `normalize`, `standard`, `log`, or ``
- `yscaler_type` **(str, default='')**: The type of y data scaler, either `minmax`, `normalize`, `standard`, `log`, or ``


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
- `method` **(str, default='gsa')**: The preliminary analysis method, including:
    - `gsa`: the pairwise global sensitivity analysis;
    - `pc`: a modified PC algorithm that include conditional indendpence test for redundancy check/filtering after gsa
- `metric` **(str, default='it-bins')**: The metric calculating the sensitivity, including:
    - `it-bins`: the information-theoretic measures (MI and CMI) using the binning approach
    - `it-knn`: the information-theoretic measures (MI and CMI) using the k-nearest-neighbor approach
    - `corr`: the correlation coefficient
- `sst` **(bool, default=False)**: Whether to perform the statistical significance test or the shuffle test
- `ntest` **(int, default=100)**: The number of shuffled samples in sst
- `alpha` **(float, default=0.05)**: The significance level used in the shuffle test
- `bins` **(int, default=10)**: The number of bins for each dimension when `"metric"` == "it-bins"
- `k` **(int, default=3)**: The number of nearest neighbors when `"metric"` == "it-knn"
- `n_jobs` **(int, default=-1)**: The number of processers/threads used by `joblib.Parallel`
- `seed_shuffle` **(int, default=1234)**: The random seed number for doing shuffle test
- `verbose` **(int, default=0)**: The verbosity level (0: normal, 1: debug)


## Configuring ensemble learning
Below are the parameters to configure the ensemble learning and passed into initializing `KIM` class instance.

```python
map_configs = {
    "map_option": "many2one",
    "mask_option":  "cond_sensitivity",
    "map_configs": map_configs,
}
```
- `map_option` **(str, default='many2one')**: The option of selecting the type of mapping, including:
    - `many2one`: the knowledge-informed mapping using the preliminary analysis result
    - `many2many`: the original mapping without being knowledge-informed
- `mask_option` **(str, default='cond_sensitivity')**: The option of which preliminary analysis result is used to mask the critial inputs used to estimate the outputs, including:
    - `sensitivity`: using the global sensitivity analysis result, $\mathbf{X}^{S_1}_j$ (using `Data.sensitivity_mask`)
    - `cond_sensitivity`: using both the global sensitivity analysis and the redundancy filtering, $\mathbf{X}^{S}_j$ (using `Data.cond_sensitivity_mask`)
- `map_configs` **(dict)**: The configurations for the mapping, including all the arguments of `kim.Map` class except x and y. See below

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
}
```
- `model_type` **(type, default='kim.mapping_model.MLP')**: The type of the mapping in [`equinox.Module`](https://github.com/patrick-kidger/equinox?tab=readme-ov-file) class
- `n_model` **(int, default=1)**: The number of ensemble models or mappings
- `ensemble_type` **(str, default='single')**: The type of ensemble learning, including
    - `single`: no ensemble with only one neural network to be trained
    - `ens_random`: generating the ensemble by performing a randomized search based on the defined hyperparameters configs in `model_hp_choices`, `optax_hp_choices`, and `dl_hp_choices`
    - `ens_grid`: generating the ensemble by performing a grid search based on the defined hyperparameters configs in `model_hp_choices`, `optax_hp_choices`, and `dl_hp_choices`
- `ens_seed` **(int, default=100)**: The random seed for generating ensemble configurations when `ensemble_type` is set to `ens_random`
- `training_parallel` **(bool, default=True)**: Whether to perform parallel training
- `parallel_config` **(dict, default=None)**: The parallel training configurations following the arguments of [`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
- `model_hp_choices` **(dict, default={})**: The tunable model hyperparameters, in dictionary format `{key: [value1, value2,...]}`. The model hyperparameters must follow the arguments of the specified `model_type`
- `model_hp_fixed` **(dict, default={})**: The fixed model hyperparameters, in dictionary format `{key: value}`. The model hyperparameters must follow the arguments of the specified `model_type`
- `optax_hp_choices` **(dict, default={})**: The tunable optimizer hyperparameters, in dictionary format `{key: [value1, value2,...]}`. The optimizer hyperparameters must follow the arguments of the specified [`optax` optimizer](https://optax.readthedocs.io/en/latest/api/optimizers.html). Hyperparameters that must be provided are `optimizer_type` (str), `nsteps` (int), and `loss_func` (callable), unless they are provided in `optax_hp_fixed`
- `optax_hp_fixed` **(dict, default={})**: The fixed optimizer hyperparameters, in dictionary format `{key: value}`. The optimizer hyperparameters must follow the arguments of the specified [`optax` optimizer](https://optax.readthedocs.io/en/latest/api/optimizers.html). Hyperparameters that must be provided are `optimizer_type` (str), `nsteps` (int), and `loss_func` (callable), unless they are provided in `optax_hp_choices`
- `dl_hp_choices` **(dict, default={})**: The tunable dataloader hyperparameters, in dictionary format `{key: [value1, value2,...]}`. The optimizer hyperparameters must follow the arguments of `kim.mapping_model.dataloader_torch.make_big_data_loader`. Hyperparameters that must be provided are  `batch_size` (int) and `num_train_sample` (int), unless they are provided in `dl_hp_fixed`
- `dl_hp_fixed` **(dict, default={})**: The fixed dataloader hyperparameters, in dictionary format `{key: value}`. The optimizer hyperparameters must follow the arguments of `kim.mapping_model.dataloader_torch.make_big_data_loader`. Hyperparameters that must be provided are  `batch_size` (int) and `num_train_sample` (int), unless they are provided in `dl_hp_fixed`