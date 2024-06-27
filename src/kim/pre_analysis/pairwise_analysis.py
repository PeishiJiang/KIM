"""Pairwise analysis using either mutual information or correlation coefficient"""

# Author: Peishi Jiang <shixijps@gmail.com>

import numpy as np

from .sst import shuffle_test


def pairwise_analysis(xdata, ydata, metric_calculator, sst=False, ntest=100, alpha=0.05):
    """Perform the pairwise analysis.

    Args:
        xdata (array-like): the predictors with shape (Ns, Nx)
        ydata (array-like): the predictands with shape (Ns, Ny)
        metric_calculator (class): the metric calculator
        sst (bool): whether to perform statistical significance test. Defaults to False.
        ntest (int): number of shuffled samples in sst. Defaults to 100.
        alpha (float): the significance level. Defaults to 0.05.

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
                metric, significance = shuffle_test(x, y, metric_calculator, ntest, alpha)
                sensitivity[i, j] = metric
                sensitivity_mask[i, j] = significance
    
    return sensitivity, sensitivity_mask

        