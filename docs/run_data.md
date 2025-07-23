# Performing the preliminary analysis

We first perform the preliminary data analysis. Depending on the configurations, this procedure potentially involves the following two steps: (1) a global sensitivity analysis and (2) redundancy check using conditional mutual information.

## Preparing the inputs and outputs
**TBD**

```python
import numpy as np

Ns = 200  # Number of samples
Nx = 10  # Number of input variables
Ny = 5  # Number of output variables

# Inputs and outputs
x = np.random.randn(Ns, Nx)
y = np.random.randn(Ns, Ny)

```

## Configuring the preliminary analysis

The detailed configurations can be found in this [post](./configs.md)

- Data configuration
```python
data_params = {
    "xscaler_type": "minmax",  # scaler for the x (input) data
    "yscaler_type": "minmax",  # scaler for the y (output) data
}
```

- Preliminary analysis configuration
```python
sensitivity_params = {
    "method": "pc", "metric": "it-knn",
    "sst": True, "ntest": 100, "alpha": 0.05, "k": 3,
    "n_jobs": 100, "seed_shuffle": 1234,
    "verbose": 1
}
```

## Run the preliminary analysis

```python
from kim import Data

data = Data(x, y, **data_params)
data.calculate_sensitivity(**sensitivity_params)
```

## Save the results

```python
f_data_save = './examples/tutorial/data'
data.save(f_data_save)
```

