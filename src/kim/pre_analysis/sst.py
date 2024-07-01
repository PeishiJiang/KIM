"""Statistical significance test or shuffle test"""

# Author: Peishi Jiang <shixijps@gmail.com>

# TODO: Use joblib to speed up the metric calculation

import numpy as np


def shuffle_test(x, y, metric_calculator, cdata=None, ntest=100, alpha=0.05, random_seed=1234):
    """Shuffle test.

    Args:
        x (array): the x data with dimension (Ns,)
        y (array): the x data with dimension (Ns,)
        cdata (array): the x data with dimension (Ns,Nc)
        metric_calculator (class): the metric calculator
        ntest (int): number of shuffled samples in sst. Defaults to 100.
        alpha (float): the significance level. Defaults to 0.05.
        random_seed (int): the random seed number. Defaults to 1234.

    Returns:
        (float, bool): metric_value, significance_or_not
    """
    # if random_seed is None:
    #     np.random.seed()
    # else:
    #     np.random.seed(random_seed)
    
    # Calculate the reference metric
    if cdata is None:
        metrics = metric_calculator(x, y)
    else:
        metrics = metric_calculator(x, y, cdata)

    # Calculate the suffled metrics
    metrics_shuffled_all = np.zeros(ntest)
    for i in range(ntest):
        # Get shuffled data
        x_shuffled = np.random.permutation(x)

        # Calculate the corresponding mi
        if cdata is None:
            metrics_shuffled = metric_calculator(x_shuffled, y)
        else:
            metrics_shuffled = metric_calculator(x_shuffled, y, cdata)

        metrics_shuffled_all[i] = metrics_shuffled

    # Calculate 95% and 5% percentiles
    upper = np.percentile(metrics_shuffled_all, int(100*(1-alpha)))
    # lower = np.percentile(metrics_shuffled_all, int(100*alpha))

    # Return
    if metrics > upper:
        return metrics, True
    else:
        return 0.0, False


# def cond_shuffle_test(x, y, xc, cond_metric_calculator, ntest=100, alpha=0.05, random_seed=1234):
#     pass