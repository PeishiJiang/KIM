"""Pairwise analysis using either mutual information or correlation coefficient"""

# Author: Peishi Jiang <shixijps@gmail.com>

import numpy as np

from .sst import shuffle_test
from .metric_calculator import MetricBase

from jaxtyping import Array


def pairwise_analysis(
    xdata: Array, ydata: Array, metric_calculator: MetricBase, sst: bool=False, 
    ntest: int=100, alpha: float=0.05, seed_shuffle: int=1234
):
    """Perform the pairwise analysis.

    Args:
        xdata (array-like): the predictors with shape (Ns, Nx)
        ydata (array-like): the predictands with shape (Ns, Ny)
        metric_calculator (class): the metric calculator
        sst (bool): whether to perform statistical significance test. Defaults to False.
        ntest (int): number of shuffled samples in sst. Defaults to 100.
        alpha (float): the significance level. Defaults to 0.05.
        seed_shuffle (int): the random seed number for doing shuffle test. Defaults to 1234.

    Returns:
        (array, array): the sensitivity result
    """
    # Data dimensions
    assert xdata.shape[0] == ydata.shape[0], \
        "xdata and ydata must be the same number of samples"
    # Ns = xdata.shape[0]
    Nx = xdata.shape[1]
    Ny = ydata.shape[1]

    # Initialize the return sensitivity values and masks
    sensitivity = np.zeros([Nx, Ny])
    sensitivity_mask = np.ones([Nx, Ny], dtype='bool')

    for i in range(Nx):
        x = xdata[:,i]
        for j in range(Ny):
            y = ydata[:,j]
            if not sst:
                sensitivity[i, j] = metric_calculator(x, y)
            else:
                metric, significance = shuffle_test(
                    x, y, metric_calculator, None, ntest, alpha, 
                    random_seed=seed_shuffle
                )
                sensitivity[i, j] = metric
                sensitivity_mask[i, j] = significance
    
    return sensitivity, sensitivity_mask

        