# Postprocessing the training results

## Loading the results
```python
from kim.map import KIM
from kim.data import Data

# Load the preliminary analysis result
f_data = './examples/tutorial/data'
data = Data(None, None)
data.load(f_data, check_xy=False)

# Load the ensemble learning result
f_kim = './examples/tutorial/kim'
kim = KIM(data, map_configs={}, mask_option="cond_sensitivity", map_option='many2one')
kim.load(f_kim_save)

# Calculate the training performances on the test dataset
results = kim.evaluate_maps_on_givendata()

```

## Plotting the preliminary analysis results
```python
from kim.utils import plot_sensitivity
# Global sensitivity analysis
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plot_sensitivity(data.sensitivity.T)
ax.set(title='Gloabal sensitivity using mutual information', xlabel='X', ylabel='Y');

```

```python
from kim.utils import plot_sensitivity_mask
# Global sensitivity + redundancy filtering check
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plot_sensitivity_mask(data.cond_sensitivity_mask.T, ylabels=x_vars, xlabels=y_vars)
ax.set(title='Global sensitivity + Redundancy filtering mask')

```

## Plotting the training results
```python
from kim.utils import plot_1to1_uncertainty
train_or_test = 'test'
fig, axes = plt.subplots(1,data.Ny,figsize=(10,6))
for i in range(data.Ny):
    ax = axes[i]
    plot_1to1_uncertainty(results, iy=i, ax=ax, train_or_test=train_or_test, y_var=i)

plt.subplots_adjust(hspace=0.4, wspace=0.4)

```
