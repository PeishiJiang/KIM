"""Preliminary analysis on the predictors X and the predictands Y"""

# Author: Peishi Jiang <shixijps@gmail.com>

import numpy as np

from .metric_calculator import get_metric_calculator
from .pairwise_analysis import pairwise_analysis
from .pc_analysis import pc

from jaxtyping import Array

def analyze_interdependency(
    xdata: Array, ydata: Array, method: str='gsa', metric: str='it-bins', 
    sst: bool=False, ntest: int=100, alpha: float=0.05, bins: int=10, 
    k: int=5, seed_shuffle: int=1234
):
    """Function for performing the interdependency between x and y.

    Args:
        xdata (array-like): the predictors with shape (Ns, Nx)
        ydata (array-like): the predictands with shape (Ns, Ny)
        method (str): The sensitivity methods, including:
            "gsa": the pairwise global sensitivity analysis
            "pc": a modified PC algorithm that include conditional indendpence test after gsa
            Defaults to 'mi-bins'.
        metric (str): The metric calculating the sensitivity, including:
            "it-bins": the information-theoretic measures (MI and CMI) using binning approach
            "it-knn": the information-theoretic measures (MI and CMI) using knn approach
            "corr": the correlation coefficient
            Defaults to 'corr'.
        sst (bool): whether to perform statistical significance test. Defaults to False.
        ntest (int): number of shuffled samples in sst. Defaults to 100.
        alpha (float): the significance level. Defaults to 0.05.
        bins (int): the number of bins for each dimension when metric == "it-knn". Defaults to 10.
        k (int): the number of nearest neighbors when metric == "it-knn". Defaults to 5.
        seed_shuffle (int): the random seed number for doing shuffle test. Defaults to 5.

    Returns:
        (array, array, array): the sensitivity result
    """
    # Data dimensions
    assert xdata.shape[0] == ydata.shape[0], \
        "xdata and ydata must be the same number of samples"
    # Ns = xdata.shape[0]
    Nx = xdata.shape[1]
    Ny = ydata.shape[1]

    # # Initialize the return sensitivity values and masks
    # sensitivity = np.zeros([Nx, Ny])
    # sensitivity_mask = np.zeros([Nx, Ny], dtype='bool')
    # cond_sensitivity_mask = np.zeros([Nx, Ny], dtype='bool')

    # Get metric calculator
    metric_calculator, cond_metric_calculator = get_metric_calculator(metric, bins, k)

    # Analyze the interdependency between xdata and ydata
    if method.lower() == "gsa":
        sensitivity, sensitivity_mask = pairwise_analysis(
            xdata, ydata, metric_calculator, sst=sst, ntest=ntest, alpha=alpha,
            seed_shuffle=seed_shuffle
        )
        # cond_sensitivity_mask = np.zeros([Nx, Ny], dtype='bool')
        cond_sensitivity_mask = sensitivity_mask

    elif method.lower() == "pc":
        sensitivity, sensitivity_mask, cond_sensitivity_mask = pc(
            xdata, ydata, metric_calculator, cond_metric_calculator, ntest=ntest, 
            alpha=alpha, seed_shuffle=seed_shuffle
        )

    else:
        raise Exception("Unknown method: %s" % method)
    
    return sensitivity, sensitivity_mask, cond_sensitivity_mask
