# Performing ensemble neural network training

Then, we leveraged the preliminary data analysis result to train the ensemble inverse mapping.

## Configuring the ensemble learning
**TBD**

The detailed configurations can be found in this [post](./configs.md)
```python
from kim.map import KIM
from kim.mapping_model import MLP

# Mapping parameters for each test below
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
        'n_jobs': 4, 
        'backend': 'loky',
        'verbose': 1
    },
    'device': None,
}
```

## Train the ensemble neural network

```python
kim_map = KIM(data, map_configs, mask_option="cond_sensitivity", map_option='many2one')
kim_map.train()
```


## Save the training results

```python
from pathlib import Path
f_kim = Path('./examples/tutorial/kim')
kim_map.save(f_kim)

```
