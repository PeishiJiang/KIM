"""Preliminary analysis on the predictors X and the predictands Y"""

# Author: Peishi Jiang <shixijps@gmail.com>

# import numpy as np

from .pairwise_analysis import pairwise_analysis
from .metric_calculator import get_metric_calculator

def analyze_interdependency(
    xdata, ydata, method='gsa', metric='it-bins', 
    sst=False, ntest=100, alpha=0.05, bins=10, k=5
):
    # # Data dimensions
    # assert xdata.shape[0] == ydata.shape[0], \
    #     "xdata and ydata must be the same number of samples"
    # Ns = xdata.shape[0]
    # Nx = xdata.shape[1]
    # Ny = ydata.shape[1]

    # # Initialize the return sensitivity values and masks
    # sensitivity = np.zeros([Nx, Ny])
    # sensitivity_mask = np.zeros([Nx, Ny], dtype='bool')

    # Get metric calculator
    metric_calculator = get_metric_calculator(metric, bins, k)

    # Analyze the interdependency between xdata and ydata
    if method.lower() == "gsa":
        sensitivity, sensitivity_mask = pairwise_analysis(
            xdata, ydata, metric_calculator, sst=sst, ntest=ntest, alpha=alpha
        )

    # elif method.lower() == "pc":
    #     sensitivity, sensitivity_mask = pc_analysis(xdata, ydata, sst=False, ntest=100, alpha=0.05)

    else:
        raise Exception("Unknown method: %s" % method)
    
    return sensitivity, sensitivity_mask
