"""parameters_initialization.py: Parameter initializer for GMM data."""

__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import numpy as np

from sklearn.cluster import KMeans


def init_kmeans(x, K):
    """Initialize parameters for EM.
    
    Args:
        x (2D numpy array): The data points.
        K (int): The number of Gaussian component.
        
    Returns:
        parameters (Dictionaries): The dictionary containing the initialized
            parameters
    """
    # Get the number of data points and the number of dimensions
    N, D = x.shape

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=K).fit(x)
    idx = kmeans.labels_
    cent = kmeans.cluster_centers_

    # Estimate parameters corresponding each component (cluster)
    pp = np.zeros([K])
    mu = np.zeros([K, D])
    sigma = np.zeros([D, D, K])

    for k in range(K):
        pp[k] = (idx==k).sum()
        mu[k, :] = cent[k, :]
        if pp[k]:
            sigma[:, :, k] = np.cov(x[idx==k, :].T)
    
    # Store the parameters
    parameters = {
        'pp' : pp / N,
        'mu' : mu,
        'sigma' : sigma
    }

    return parameters