"""likelihood_estimation.py: Likelihood estimators for truncated and censored data."""

__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import numpy as np

from scipy.stats import mvn, norm
from scipy.stats import multivariate_normal


def estimate_censored_loglikelihood_component(x, pp, mu, sigma, pattern):
    """ Estimate unnormalized weighted log likelihood of censored dataset 
        corresponding to a particular Gaussian component.
    
    Args:
        x (2D numpy array): The observed data.
        pp (float): The mixing weight.
        mu (1D numpy array): The mean of the Gaussian component.
        sigma (2D numpy array): The covariance matrix of the Gaussian component.
        pattern (dictionary): The censoring pattern of the dataset.
        
    Returns:
        log_likelihood (1D numpy array): The log likelihood of each data point.
    """
    eps = 10e-16

    # Extract the shape of data and the censoring patterns
    N = x.shape[0]
    unique_pattern = pattern['uniq_censored']
    num_patterns = pattern['count']
    same_pattern = pattern['xpttn']
    integral_regions = pattern['range']

    # Compute the logarithm of component probability
    log_p = np.log(pp)

    # Conduct mean substraction on the data
    x0 = x - mu

    # Loop over each pattern and then estimate the corresponding log likelihood
    # of censored elements
    log_likelihood = np.zeros([N, 1])
    for ii in range(num_patterns.shape[0]):
        on = unique_pattern[ii, :] # The uncensored index
        mn = ~on                   # The censored index
        do = np.sum(on)            # The number of censored dimensions
        dm = np.sum(mn)            # The number of uncensored dimensions
        idx = np.where(same_pattern[:, ii])[0]

        # Compute log likelihood of observed x_o
        # Factorize the covariance matrix by using Cholesky decomposition
        R = np.linalg.cholesky(sigma[on, :][:, on]).T

        # Compute the quadform term
        x_Rinv = x0[idx, :][:, on].dot(np.linalg.inv(R))
        quadform = np.sum(x_Rinv**2, axis=1).reshape([-1, 1])

        # Compute the logarithm of the determinant of covariance matrix
        log_sqrt_detsigma = np.sum(np.log(R.diagonal()))

        # Check if existing an censored dimension
        if np.sum(mn):
            # Estimate the mean and covariance of the conditional 
            # normal distribution
            mu_mo = np.dot((x0[idx, :][:, on].dot(np.linalg.inv(R))),
                           (sigma[mn, :][:, on].dot(np.linalg.inv(R))).T)
            mu_mo = mu_mo + mu[mn]
            sigma_Rinv = sigma[mn, :][:, on].dot(np.linalg.inv(R))
            sigma_mo = sigma[mn, :][:, mn] - sigma_Rinv.dot(sigma_Rinv.T)

            # Numerical stability
            zero_error = np.diag(sigma_mo) < eps
            if np.sum(zero_error):
                sigma_mo[np.diag(sigma_mo)] = eps
            
            # Compute the multivariate normal CDF evaluated over the rectangle
            # with  lower and upper bounds
            integral_idx = [integral_regions[i] for i in idx.tolist()]
            integral_low = np.array([integral_idx[i][0] \
                for i in range(len(integral_idx))])
            integral_up = np.array([integral_idx[i][1] \
                for i in range(len(integral_idx))])

            phi = np.zeros([integral_low.shape[0], 1])
            for i in range(integral_low.shape[0]):
                phi[i] = mvn.mvnun((integral_low - mu_mo)[i], 
                                   (integral_up - mu_mo)[i], 
                                   np.squeeze(np.zeros([1, dm])), 
                                   sigma_mo)[0]

            log_phi = np.log(phi)
        else:
            log_phi = 0


        # Compute weighted log likelihood of x
        log_likelihood[idx] = ((-0.5*quadform) + (-float(log_sqrt_detsigma) +\
             float(log_p) + log_phi) - float((do * np.log(2*np.pi) / 2.)))
    
    return log_likelihood.reshape([-1])

def estimate_censored_loglikelihood(x, pp, mu, sigma, pattern):
    """Estimate the log likelihood of a censored GMM dataset.
    
    Args:
        x (2D numpy array): The data points.
        pp (1D numpy array): The mixing weights.
        mu (2D numpy array): The means of the Gaussian components.
        sigma (3D numpy array): The covariance matrices of the Gaussian 
            components.
        pattern (dictionary): The censoring pattern of the dataset.
        
    Returns:
        Returns:
        post (1D numpy array): The estimated posterior probabilities.
        log_likelihood(1D numpy array): The normalized log-likelihood.
        total_log_likelihood (float): The total log likelihood over the dataset.
    """ 
    N = x.shape[0]
    K = mu.shape[0]

    log_likelihood = np.zeros([N, K])

    # Compute log likelihood of each Gaussian component
    for k in range(K):
        log_likelihood[:, k] = estimate_censored_loglikelihood_component(
            x, pp[k], mu[k, :], sigma[:, :, k], pattern)
    
    # Replace nan by -inf to avoid numerical issue
    log_likelihood[np.isnan(log_likelihood)] = -np.inf
    
    # Estimate the posterior probabilities
    max_ll = (np.max(log_likelihood, axis=1)).reshape([-1, 1])
    post = np.exp(log_likelihood - max_ll)

    # Estimate the density and then normalize the posterior
    density = np.sum(post, axis=1).reshape([-1, 1])
    post = post / density

    # Normalize the log-likelihood
    log_likelihood = np.log(density) + max_ll

    # Compute the total log-likelihood
    total_log_likelihood = np.sum(log_likelihood)
    
    return post, log_likelihood, total_log_likelihood

def estimate_truncated_censored_loglikelihood(x, pp, mu, sigma, 
                                              pattern, truncated_bounds):
    """Estimate the log likelihood of a truncated and censored data points.
    
    Args:
        x (2D numpy array): The observed data points.
        pp (1D numpy array): The mixing weights.
        mu (2D numpy array): The means of the Gaussian components.
        sigma (3D numpy array): The covariance matrices of the 
            Gaussian components.
        pattern (dictionary): The censoring pattern of the dataset.
        truncated_bounds (1D numpy array): The lower and upper bounds.
    Returns:
        post: (1D numpy array): The estimated posterior probabilities.
        truncated_censored_loglikelihood (float): The estimated log-likelihood.
    """ 
    N = x.shape[0]
    K = mu.shape[0]
    
    # Estimate the log likelihood of censored data
    post, _, censored_log_likelihood = estimate_censored_loglikelihood(
        x, pp, mu, sigma, pattern)
    
    # Estimate the log likelihood of truncated and censored data
    cdf = norm.cdf(truncated_bounds[0], mu[:, 0], 
                   np.sqrt(np.squeeze(sigma[0, 0, :])))
    truncated_censored_loglikelihood = censored_log_likelihood -\
        N * np.log(pp.dot((1 - cdf)))
    
    return post, truncated_censored_loglikelihood