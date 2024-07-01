"""The metric calculator"""

# Author: Peishi Jiang <shixijps@gmail.com>

import numpy as np
from numpy import ma
from scipy.spatial import cKDTree
from scipy.special import digamma


def get_metric_calculator(metric="corr", bins=10, k=5):

    if metric.lower() == "corr":
        metric_calculator = CorrCoef()
        cond_metric_calculator = ParCorrCoef()
    elif metric.lower() == "it-bins":
        metric_calculator = MIbins(bins)
        cond_metric_calculator = CMIbins(bins)
    elif metric.lower() == "it-knn":
        metric_calculator = MIknn(k)
        cond_metric_calculator = CMIknn(k)
    
    return metric_calculator, cond_metric_calculator


class CorrCoef(object):
    """Correlation coefficient"""

    def __init__(self):
        self.metric = "corr"

    def __call__(self, x, y) -> float:
        return np.corrcoef(x, y)[0,1] 


class ParCorrCoef(object):
    """Conditional correlation coefficient"""

    def __init__(self):
        self.metric = "corr"

    def __call__(self, x, y, cdata) -> float:
        # This implementation is inspired by the following code:
        # https://github.com/jakobrunge/tigramite/blob/4a6a470eaa67bc57a827b2f70b26ef35650ffcdc/tigramite/independence_tests/parcorr.py#L124
        xresiduals = self._get_ols_residuals(x, cdata)
        yresiduals = self._get_ols_residuals(y, cdata)
        return np.corrcoef(xresiduals, yresiduals)[0, 1]
    
    def _get_ols_residuals(self, data, cdata) -> float:
        # Perform OLS
        beta_hat = np.linalg.lstsq(cdata, data, rcond=None)[0]
        data_est = np.dot(cdata, beta_hat)
        # Calculate the residuals
        residuals = data - data_est
        return residuals


class MIbins(object):
    """Mutual information using the binning method"""

    def __init__(self, bins=10):
        self.metric = "it-bins"
        self.bins = bins

    def __call__(self, x, y) -> float:
        return computeMIbins(x, y, self.bins)


class CMIbins(object):
    """Conditional mutual information using the binning method"""

    def __init__(self, bins=10):
        self.metric = "it-bins"
        self.bins = bins

    def __call__(self, x, y, cdata) -> float:
        return computeCMIbins(x, y, cdata, self.bins)


class MIknn(object):
    """Mutual information using the k-nearest-neighbor method"""

    def __init__(self, k=10):
        self.metric = "it-knn"
        self.k = k

    def __call__(self, x, y) -> float:
        return computeMIknn(x, y, self.k)


class CMIknn(object):
    """Conditional mutual information using the k-nearest-neighbor method"""

    def __init__(self, k=10):
        self.metric = "it-knn"
        self.k = k

    def __call__(self, x, y, cdata) -> float:
        return computeCMIknn(x, y, cdata, self.k)


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


def computeMIbins(x, y, bins=10) -> float:
    """Compute the mutual information I(X;Y) using the binning method.

    Args:
        x (array): the x data with dimension (Ns,)
        y (array): the y data with dimension (Ns,)
        bins (int): the number of bins for each dimension in the probability calculation. Defaults to 10.
    Returns:
        float: the mutual information
    """
    # Compute the entropies
    hxy = computeEntropybins(np.array([x, y]).T, bins)
    hx  = computeEntropybins(np.expand_dims(x, 1), bins)
    hy = computeEntropybins(np.expand_dims(y, 1), bins)

    # Compute the mutual information
    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = hx + hy - hxy

    return mi


def computeCMIbins(x, y, cdata, bins=10) -> float:
    """Compute the conditional mutual information I(X;Y|C) using the binning method.

    Args:
        x (array): the x data with dimension (Ns,)
        y (array): the y data with dimension (Ns,)
        cdata (array): the conditional data with dimension (Ns, Nc)
        bins (int): the number of bins for each dimension in the probability calculation. Defaults to 10.
    Returns:
        float: the conditional mutual information
    """
    x = np.expand_dims(x, 1)  # (Ns, 1)
    y = np.expand_dims(y, 1)  # (Ns, 1)

    # Compute the entropies
    hxyc = computeEntropybins(np.concat([x, y, cdata], axis=1), bins)
    hxc  = computeEntropybins(np.concat([x, cdata], axis=1), bins)
    hyc  = computeEntropybins(np.concat([y, cdata], axis=1), bins)
    hc   = computeEntropybins(cdata, bins)

    # Compute the mutual information
    # I(X;Y|C) = H(X,C) + H(Y,C) - H(X,Y,C) - H(C)
    cmi = hxc + hyc - hxyc - hc

    return cmi


def computeMIknn(x, y, k=2) -> float:
    """Compute the  mutual information I(X;Y) using the k-nearest-neighbor method,
       based on the original formula (not the average version).
       Modified from: https://github.com/PeishiJiang/info/blob/master/info/core/info.py#L1315.

    Args:
        x(array): the x data with dimension (Ns,)
        y(array): the y data with dimension (Ns,)
        k (int): the nearest neighbor. Defaults to 2.
    Returns:
        float: the mutual information
    """
    assert x.shape[0] == y.shape[0], \
        "xdata and ydata must be the same number of samples"
    
    npts = x.shape[0]

    data = np.array([x, y]).T
    x= np.expand_dims(x, 1)
    y= np.expand_dims(y, 1)

    # Compute the ball radius of the k nearest neighbor for each data point
    tree = cKDTree(data)
    dist, ind = tree.query(data, k+1, p=float('inf'))
    rset    = dist[:, -1][:, np.newaxis]

    # Locate the index where rset are zero, and change these values to 1e-14
    rset[rset == 0] = 1e-14

    # Get the number of nearest neighbors for X and Y based on the ball radius
    treey, treex = cKDTree(y), cKDTree(x)
    kyset = np.array([len(treey.query_ball_point(y[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])
    kxset = np.array([len(treex.query_ball_point(x[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])

    # Compute information metrics
    return digamma(npts) + digamma(k) - np.mean(digamma(kyset)) - np.mean(digamma(kxset))


def computeCMIknn(x, y, cdata, k=2) -> float:
    """Compute the conditional mutual information I(X;Y|C) using the k-nearest-neighbor method,
       based on the original formula (not the average version).
       Modified from: https://github.com/PeishiJiang/info/blob/master/info/core/info.py#L1315.

    Args:
        x(array): the x data with dimension (Ns,)
        y(array): the y data with dimension (Ns,)
        cdata(array): the conditional data with dimension (Ns,Nc)
        k (int): the nearest neighbor. Defaults to 2.
    Returns:
        float: the conditional mutual information
    """
    # assert x.shape[0] == y.shape[0], \
    #     "x and y must be the same number of samples"
    # assert x.shape[0] == cdata.shape[0], \
    #     "x and c must be the same number of samples"
    
    npts = x.shape[0]

    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    data = np.concat([x, y, cdata], axis=1)
    xcdata = np.concat([x, cdata], axis=1)
    ycdata = np.concat([y, cdata], axis=1)

    # Compute the ball radius of the k nearest neighbor for each data point
    tree = cKDTree(data)
    dist, ind = tree.query(data, k+1, p=float('inf'))
    rset    = dist[:, -1][:, np.newaxis]

    # Locate the index where rset are zero, and change these values to 1e-14
    rset[rset == 0] = 1e-14

    # Get the number of nearest neighbors for X and Y based on the ball radius
    # treey, treex = cKDTree(y), cKDTree(x)
    # kyset = np.array([len(treey.query_ball_point(y[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])
    # kxset = np.array([len(treex.query_ball_point(x[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])
    treeyc, treexc, treec = cKDTree(ycdata), cKDTree(xcdata), cKDTree(cdata)
    kycset = np.array([len(treeyc.query_ball_point(ycdata[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])
    kxcset = np.array([len(treexc.query_ball_point(xcdata[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])
    kcset  = np.array([len(treec.query_ball_point(cdata[i,:], rset[i]-1e-15, p=float('inf'))) for i in range(npts)])

    # Compute information metrics
    return np.mean(digamma(kcset)) + digamma(k) - np.mean(digamma(kycset)) - np.mean(digamma(kxcset))
    # return digamma(npts) + digamma(k) - np.mean(digamma(kyset)) - np.mean(digamma(kxset))