"""standard_em.py: Implementation of standard EM algorithm."""

__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import numpy as np

from sklearn import mixture


def perform_standard_em(x, K, seed):
    """Estimate GMM's parameters by using the standard EM algorithm.

    Args:
        x (2D numpy array): The observed data.
        K (int): The number of mixture component.
        seed (int): The random seed.

    Returns:
        results (dictionary): The dictionary containing estimated parameters.
    """
    model = mixture.GaussianMixture(n_components=K, random_state=seed)
    model.fit(x)

    results = {
        'K': K,
        'pp': model.weights_,
        'mu': model.means_,
        'sigma': model.covariances_.T
    }

    return results
