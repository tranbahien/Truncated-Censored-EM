"""moments_estimation.py: Utilities used to esttimate moments."""

__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import numpy as np

from scipy.stats import mvn, norm
from scipy.stats import multivariate_normal


def compute_univariate_marginal(x, k, sigma, a, b):
    """Compute the univariate marginal of a normally distributed dataset.
    
    Args:
        x (1D numpy array): The k-th dimension of the dataset.
        sigma (2D numpy array): The covariance matrix of the density function.
        k (int): The index of the target dimension.
        a (1D numpy array): The truncation lower bound.
        b (1D numpy array): The truncation upper bound.
        
    Returns:
        Fk (1D numpy array): The calculated first moment.
    """
    # Basic initialization
    N = x.shape[0]
    D = sigma.shape[0]
    Fk = np.zeros([N]) 

    # CASE 1: If x = inf or -inf, then Fk = 0
    # Find infinity elements
    idx = ~np.isinf(x) 
    if np.sum(~idx):
        Fk[~idx] = 0

    if (np.sum(idx) == 0):
        return Fk

    # CASE 2: Check if the dataset is univariate (D = 1)
    if D == 1:
        Fk[idx] = norm.pdf(x[idx], 0, np.sqrt(float(sigma)))
        return Fk

    # CASE 3: Consider the case of multivariate dataset (D > 1)    
    o = np.zeros([D], dtype=np.bool)
    o[k] = True
    m = ~o

    # Get the mean and covarance excepted the k-th dimension
    cmu = (sigma[m, :][:, o] / sigma[o, :][:, o] * x[idx]).T
    csig = sigma[m, :][:, m] - sigma[m, :][:, o] / sigma[o, :][:, o] *\
        sigma[o, :][:, m]

    # Estimate the value of cumulative distribution function for each dimension
    cdf = np.zeros([a[idx, :].shape[0], 1])
    for i in range(a[idx, :].shape[0]):
        cdf[i] = mvn.mvnun((a[idx, :][:, m] - cmu)[i], 
              (b[idx, :][:, m] - cmu)[i], 
              np.zeros([D-1]), np.squeeze(csig))[0]
    
    # Calculate the the first moment for each dimension
    Fk[idx] = (norm.pdf(x[idx], 0, np.squeeze(np.sqrt(sigma[o, :][:, o]))).\
        reshape([-1, 1]) * cdf).reshape([-1])

    return Fk

def compute_bivariate_marginal(xk, xq, k, q, sigma, a, b):
    """Compute the bivariate marignal of a normally distributed dataset.
    
    Args:
        xk (1D numpy array): The k-th dimension of the dataset.
        xq (1D numpy array): The q-th dimension of the dataset.
        k (int): The index of the first target dimension.
        q (int): The index of the second target dimension.
        sigma (2D numpy array): The covariance matrix of the density function.
        a (1D numpy array): The truncation lower bound.
        b (1D numpy array): The truncation upper bound.
        
    Returns:
        Fkq (1D numpy array): The calculated first moment.
    """
    # Basic initialization
    N = xk.shape[0]
    D = sigma.shape[0]
    Fkq = np.zeros([N])
    
    # CASE 1: If xk or xq = inf or -inf, then Fkq = 0

    # Find infinity elements
    idx = ~(np.isinf(xk) | np.isinf(xq))
    if np.any(~idx):
        Fkq[~idx] = 0
        
    if ~np.any(idx):
        return Fkq
    
    # CASE 2: Check if the dataset is univariate (D = 1). If so, Fkq = 0.
    if D == 1:
        Fkq = 0
        return Fkq
    
    # CASE 3: Consider the case of bivariate dataset

    # Change the order of variables
    if k < q:
        x = np.stack([xk, xq], axis=1)
    else:
        x = np.stack([xq, xk], axis=1)
        
    if D == 2:
        Fkq[idx] = multivariate_normal.pdf(x[idx, :], [0, 0], np.squeeze(sigma))
        return Fkq


def compute_multivariate_first_moment(x, sigma, a, b):
    """Compute the multivariate first moment of a normally distributed dataset.
    
    Args:
        x (1D numpy array): The data points.
        sigma (2D numpy array): The covariance matrix of the density function.
        a (1D numpy array): The truncation lower bound.
        b (1D numpy array): The truncation upper bound.
        
    Returns:
        Fk (1D numpy array): The calculated first moment.
    """
    # Initialization
    N, D = x.shape
    Fk = np.zeros([N, D])

    # Compute the partial first moment for each dimension
    for k in range(D):
        Fk[:, k] = compute_univariate_marginal(x[:, k], k, sigma, a, b)

    return Fk


def compute_multivariate_second_moment(xk, xq, sigma, a, b):
    """Compute the multivariate first moment of a normally distributed dataset.
    
    Args:
        xk (1D numpy array): The dataset excluded the k-th dimension.
        xq (1D numpy array): The dataset excluded the q-th dimension.
        sigma (2D numpy array): The covariance matrix of the density function.
        a (1D numpy array): The truncation lower bound.
        b (1D numpy array): The truncation upper bound.
        
    Returns:
        Fkq (1D numpy array): The calculated first moment.
    """
    # Initialization
    N, D = xk.shape
    Fkq = np.zeros([D, D, N])
    
    # Compute the partial second moment for each pair of dimensions
    for k in range(D):
        for q in range(D):
            if q != k:
                Fkq[k, q, :] = compute_bivariate_marginal(
                    xk[:, k], xq[:, q], k, q, sigma, a, b)

    return Fkq
