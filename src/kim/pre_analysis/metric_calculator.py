"""The metric calculator"""

# Author: Peishi Jiang <shixijps@gmail.com>

import numpy as np
from numpy import ma
from scipy.spatial import cKDTree
from scipy.special import digamma


def get_metric_calculator(metric="corr", bins=10, k=5):

    if metric.lower() == "corr":
        metric_calculator = CorrCoef()
    elif metric.lower() == "it-bins":
        metric_calculator = MIbins(bins)
    elif metric.lower() == "it-knn":
        metric_calculator = MIknn(k)
    
    return metric_calculator


class CorrCoef(object):
    """Correlation coefficient"""

    def __init__(self):
        self.metric = "corr"

    def __call__(self, xdata, ydata) -> float:
        return np.corrcoef(xdata, ydata)[0,1] 


class MIbins(object):
    """Mutual information using the binning method"""

    def __init__(self, bins=10):
        self.metric = "it-bins"
        self.bins = bins

    def __call__(self, xdata, ydata) -> float:
        return computeMIbins(xdata, ydata, self.bins)


class MIknn(object):
    """Mutual information using the k-nearest-neighbor method"""

    def __init__(self, k=10):
        self.metric = "it-knn"
        self.k = k

    def __call__(self, xdata, ydata) -> float:
        return computeMIknn(xdata, ydata, self.k)


def computeEntropybins(data, bins):
    """Compute the entropy using the binning method.

    Args:
        data (array): the x data with dimension (Ns,Nd)
        bins (int): the number of bins for each dimension in the probability calculation. Defaults to 10.
    Returns:
        float: the entropy
    """
    # Compute the histogram
    pdf, _ = np.histogramdd(data, bins=bins, density=False)
    pdf = pdf / pdf.sum()

    # Compute the entropy
    log_pdf = ma.filled(np.log(ma.masked_equal(pdf, 0)), 0)
    ent = -np.sum(pdf*log_pdf)
    
    return ent


def computeMIbins(xdata, ydata, bins=10) -> float:
    """Compute the mutual information I(X;Y) using the binning method.

    Args:
        xdata (array): the x data with dimension (Ns,)
        ydata (array): the y data with dimension (Ns,)
        bins (int): the number of bins for each dimension in the probability calculation. Defaults to 10.
    Returns:
        float: the mutual information
    """
    pass
    # Compute the entropies
    h12 = computeEntropybins(np.array([xdata, ydata]).T, bins)
    h1  = computeEntropybins(np.expand_dims(xdata, 1), bins)
    h2  = computeEntropybins(np.expand_dims(ydata, 1), bins)

    # Compute the mutual information
    # I(X1;X2) = H(X1) + H(X2) - H(X1,X2)
    mi = h1 + h2 - h12

    return mi


def computeMIknn(xdata, ydata, k=2) -> float:
    """Compute the  mutual information I(X;Y) using the k-nearest-neighbor method,
       based on the original formula (not the average version).
       Modified from: https://github.com/PeishiJiang/info/blob/master/info/core/info.py#L1315.

    Args:
        xdata (array): the x data with dimension (Ns,)
        ydata (array): the y data with dimension (Ns,)
        k (int): the nearest neighbor. Defaults to 2.
    Returns:
        float: the mutual information
    """
    assert xdata.shape[0] == ydata.shape[0], \
        "xdata and ydata must be the same number of samples"
    
    npts = xdata.shape[0]

    data = np.array([xdata, ydata]).T
    xdata = np.expand_dims(xdata, 1)
    ydata = np.expand_dims(ydata, 1)

    # Compute the ball radius of the k nearest neighbor for each data point
    tree = cKDTree(data)
    dist, ind = tree.query(data, k+1, p=float('inf'))
    rset    = dist[:, -1][:, np.newaxis]

    # Locate the index where rset are zero, and change these values to 1e-14
    rset[rset == 0] = 1e-14

    # Get the number of nearest neighbors for X and Y based on the ball radius
    treey, treex = cKDTree(ydata), cKDTree(xdata)
    kyset = np.array([len(treey.query_ball_point(ydata[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])
    kxset = np.array([len(treex.query_ball_point(xdata[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])

    # Compute information metrics
    return digamma(npts) + digamma(k) - np.mean(digamma(kyset)) - np.mean(digamma(kxset))