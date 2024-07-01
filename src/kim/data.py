"""The general data class."""

# Author: Peishi Jiang <shixijps@gmail.com>

import numpy as np

from .pre_analysis import analyze_interdependency


class Data(object):
    """Data object.

    Arguments:
    ----------
    xdata (array-like): the predictors with shape (Ns, Nx)
    ydata (array-like): the predictands with shape (Ns, Ny)

    Attributes
    ----------
    self.xdata (array-like): the copy of xdata
    self.ydata (array-like): the copy of ydata
    self.Ns (int): the number of samples
    self.Nx (int): the number of predictors
    self.Ny (int): the number of predictands
    self.sensitivity_config (dict): the sensitivity analysis configuration
    self.sensitivity_done (bool): whether the sensitivity analysis is performed
    self.sensitivity (array-like): the calculated sensitivity with shape (Nx, Ny)
    self.sensitivity_mask (array-like): the calculated sensitivity mask with shape (Nx, Ny)
    self.cond_sensitivity_mask (array-like): the calculated conditional sensitivity mask with shape (Nx, Ny)

    """

    def __init__(self, xdata, ydata):
        # Data array
        self.xdata = xdata
        self.ydata = ydata

        # Data dimensions
        assert xdata.shape[0] == ydata.shape[0], \
            "xdata and ydata must be the same number of samples"
        self.Ns = xdata.shape[0]
        self.Nx = xdata.shape[1]
        self.Ny = ydata.shape[1]

        # Data sensitivity
        self.sensitivity_config = {
            "method": None,
            "metric": None,
            "sst": None,
            "ntest": None,
            "alpha": None,
            "bins": None,
            "k": None,
        }
        self.sensitivity = np.zeros([self.Nx, self.Ny])
        self.sensitivity_mask = np.zeros([self.Nx, self.Ny], dtype='bool')
        self.cond_sensitivity_mask = np.zeros([self.Nx, self.Ny], dtype='bool')
        self.sensitivity_done = False
    

    def calculate_sensitivity(
        self, method='gsa', metric='it-bins', 
        sst=False, ntest=100, alpha=0.05, 
        bins=10, k=5
    ):
        """Calculate the sensitivity between xdata and ydata.

        Args:
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
        """
        sensitivity_config = self.sensitivity_config
        xdata, ydata = self.xdata, self.ydata
        # (TODO) Calculate sensitivity
        sensitivity, sensitivity_mask, cond_sensitivity_mask = analyze_interdependency(
            xdata, ydata, method, metric, sst, ntest, alpha, bins, k
        )

        # Update the configuration
        sensitivity_config['method'] = method
        sensitivity_config['metric'] = metric
        sensitivity_config['sst'] = sst
        sensitivity_config['ntest'] = ntest
        sensitivity_config['alpha'] = alpha
        sensitivity_config['bins'] = bins
        sensitivity_config['k'] = k
        self.sensitivity_config = sensitivity_config

        # Update the analysis result
        self.sensitivity_done = True
        self.sensitivity = sensitivity
        self.sensitivity_mask = sensitivity_mask
        self.cond_sensitivity_mask = cond_sensitivity_mask