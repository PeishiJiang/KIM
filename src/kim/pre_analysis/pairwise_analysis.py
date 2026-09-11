# Pairwise analysis using either mutual information or correlation coefficient
#
# Author: Peishi Jiang <shixijps@gmail.com>

import numpy as np
from joblib import Parallel, delayed

from .sst import shuffle_test
from .metric_calculator import MetricBase
from tqdm import tqdm

from jaxtyping import Array


def pairwise_analysis(
    xdata: Array, ydata: Array, metric_calculator: MetricBase, sst: bool=False, 
    ntest: int=100, alpha: float=0.05, n_jobs: int=-1, seed_shuffle: int=1234, verbose: int=0
):
    """Perform the pairwise analysis using either mutual information or correlation coefficient.

    Args:
        xdata (array-like): the predictors with shape (Ns, Nx)
        ydata (array-like): the predictands with shape (Ns, Ny)
        metric_calculator (class): the metric calculator
        sst (bool): whether to perform statistical significance test. Defaults to False.
        ntest (int): number of shuffled samples in sst. Defaults to 100.
        alpha (float): the significance level. Defaults to 0.05.
        n_jobs (int): the number of processers/threads used by joblib. Defaults to -1.
        seed_shuffle (int): the random seed number for doing shuffle test. Defaults to 1234.
        verbose (int): the verbosity level (0: normal, 1: debug). Defaults to 0.

    Returns:
        (array, array): the sensitivity, the sensitivity mask
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

    if verbose == 1:
        print("Performing pairwise analysis to remove insensitive inputs ...")

    def analyze_one_input(x):
        # Sensitivity of one input x to every output, computed serially in one worker.
        # shuffle_test reseeds from seed_shuffle on every call, so the result is
        # independent of which worker runs it and identical to the serial version.
        sens_i = np.zeros(Ny)
        mask_i = np.ones(Ny, dtype='bool')
        for j in range(Ny):
            y = ydata[:,j]
            if not sst:
                sens_i[j] = metric_calculator(x, y)
            else:
                sens_i[j], mask_i[j] = shuffle_test(
                    x, y, metric_calculator, None, ntest, alpha, 
                    n_jobs=1, random_seed=seed_shuffle
                )
        return sens_i, mask_i

    # Parallelize over the Nx inputs. Each task (Ny shuffle tests of ~100 metric
    # evaluations) is large enough to amortize the dispatch cost, unlike the
    # previous scheme of one joblib call of ntest tiny tasks per (x, y) pair.
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(analyze_one_input)(xdata[:,i]) for i in tqdm(range(Nx))
    )
    for i, (sens_i, mask_i) in enumerate(results):
        sensitivity[i, :] = sens_i
        sensitivity_mask[i, :] = mask_i
    
    return sensitivity, sensitivity_mask

        