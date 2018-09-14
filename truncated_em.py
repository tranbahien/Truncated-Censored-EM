"""truncated_em.py: Implementaion of truncated and censored EM algorithm."""

__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import numpy as np

from scipy.stats import mvn, norm
from scipy.stats import multivariate_normal

from censored_data_utils import find_censoring_pattern
from likelihood_estimation import estimate_truncated_censored_loglikelihood
from moments_estimation import compute_multivariate_first_moment, \
    compute_multivariate_second_moment

        
def estimate_mean_cov_truncated_normal(mu, sigma, a_star, b_star):
    """Estimate the mean and covariance of a truncated normal distribution
    
    Args:
        mu (1D numpy array): The mean of the conditional distribution y|x.
        sigma (2D numpy array): The mean of the conditional distribution y|x.
        a_star (1D numpy array): The truncation lower bound.
        b_star (1D numpy array): The truncation upper bound.
        
    Returns:
        tmu (1D numpy array): The estimated mean of the truncated distribution.
        tcov (2D numpy array): The estimated covariance of the truncated 
            distribution.
        alpha (1D numpy array): The normalization constant which is the normal
            integration over rectangle with vertices a star and b star,
            P(a_star < X < b_star).
    """
    # Get the numbers of datapoints and the dimensions
    N = a_star.shape[0]
    D = mu.shape[1]

    # Get the true censoring bounds
    a = a_star - mu
    b = b_star - mu

    # Avoid numerical issues by replacing infinity by zero
    an = a.copy()
    bn = b.copy()
    an[np.isinf(a)] = 0
    bn[np.isinf(b)] = 0

    # Compute the normalize constant alpha
    alpha = np.zeros([a.shape[0], 1])
    for i in range(a.shape[0]):
        alpha[i] = mvn.mvnun(a[i], b[i], np.squeeze(np.zeros([1, D])), sigma)[0]
        
    # Evaluate the first moment at the lower and upper bounds
    Fa = compute_multivariate_first_moment(a, sigma, a, b)
    Fb = compute_multivariate_first_moment(b, sigma, a, b)    
    
    # Estimate the expected value of observed data X
    tEX = (Fa - Fb).dot(sigma)
    tEX = tEX / np.matlib.repmat(alpha, 1, 1)

    # Estimate the mean of the truncated data
    tmu = tEX + mu

    # Evaluate the second moment at the vertices of the rectangle
    Faa = compute_multivariate_second_moment(a, a, sigma, a, b)
    Fab = compute_multivariate_second_moment(a, b, sigma, a, b)
    Fba = compute_multivariate_second_moment(b, b, sigma, a, b)
    Fbb = compute_multivariate_second_moment(b, b, sigma, a, b)

    # Now, let's estimate the covariance of the truncated data 
    tEXX = np.zeros([D, D, N])
    tcov = np.zeros([D, D, N])

    F1 = an * Fa - bn * Fb
    F2 = Faa + Fbb - Fab - Fba
    
    # Consider each data point
    for n in range(N):
        tEXX[:, :, n] = alpha[n] * sigma + sigma.dot(np.diag(F1[n]) /\
            np.diag(sigma.T)).dot(sigma) + sigma.dot(F2[:, :, n] -\
            np.diag(np.diag(F2[:, :, n].dot(sigma)) /\
            np.diag(sigma))).dot(sigma)

        tEXX[:, :, n] = tEXX[:, :, n] / alpha[n]

        tcov[:, :, n] = tEXX[:, :, n] - (tEX[n, :].reshape([-1, 1])).\
        dot(tEX[n, :].reshape([1, -1]))
        
    return tmu, tcov, alpha

def estimate_sufficient_statistics(x, mu, sigma, pattern):
    """Estimate the sufficient statistics of the truncated EM given observed 
        data and initialized parameters.
    
    Args:
        x (2D numpy array): The observed data.
        mu (2D numpy array): The initialized means.
        sigma (3D numpy array): The initialized covariance.
        pattern (dictionary): The truncated and censoring patterns of the data.
        
    Returns:
        xhat (2D numpy array): The estimation of the uncensored data (y).
        Q (3D numpy array): The estimation of the covariance of the uncensored
            data (y).
        alpha (1D numpy array): The probability mass in the truncated region.
    """
    # Basic initialization
    N, D = x.shape
    x0 = x - mu
    x_hat = x.copy()
    Q = np.zeros([D, D, N])
    alpha = np.ones(N)

    # Get the statistics regarding patterns of the censored data
    unique_pattern = pattern['uniq_censored']
    num_patterns = pattern['count']
    same_pattern = pattern['xpttn']
    integral_regions = pattern['range']

    # Consider independently each pattern 
    for ii in range(len(num_patterns)):
        # Get the mask of censored (mn) and uncensored (on) components
        on = unique_pattern[ii, :]
        mn = ~on
        idx = np.where(same_pattern[:, ii])[0]

        # Check the case of fully observed data
        if np.all(on):
            continue

        # Factorize the covariance matrix by using Cholesky decomposition
        R = np.linalg.cholesky(sigma[on, :][:, on]).T

        # Compute the mean and covariance of the conditional normal distribution
        mu_mo = np.dot((x0[idx, :][:, on].dot(np.linalg.inv(R))),
                       (sigma[mn, :][:, on].dot(np.linalg.inv(R))).T)
        mu_mo = mu_mo + mu[mn]

        sigma_Rinv = sigma[mn, :][:, on].dot(np.linalg.inv(R))
        sigma_mo = sigma[mn, :][:, mn] - sigma_Rinv.dot(sigma_Rinv.T)

        # Estimate the mean and covariance of the truncated normal distribution
        integral_idx = [integral_regions[i] for i in idx.tolist()]
        integral_low = np.array([integral_idx[i][0] \
            for i in range(len(integral_idx))])
        integral_up = np.array([integral_idx[i][1]\
            for i in range(len(integral_idx))])
        tmu, tcov, talpha = estimate_mean_cov_truncated_normal(
            mu_mo, sigma_mo, integral_low, integral_up)
        
        # Replace censored elements by the estimated mean
        mn_indices = np.where(mn == True)[0]
        if np.sum(mn) == 1:
            x_hat[idx, mn_indices[0]] = tmu.reshape([-1])
        else:
            tmu = tmu.reshape(x_hat[idx, :][:, mn].shape)
            for mn_idx in mn_indices:
                x_hat[idx, mn_idx] = tmu[:, mn_idx]

        # Perform covariance corrections for censored elements
        tcov.reshape(Q[mn,:,:][:,mn,:][:,:,idx].shape)
        
        if np.sum(mn) == 1:
            Q[mn_indices[0], mn_indices[0], idx] = tcov
        
        else:
            for r_mn_idx, r_cov_idx in \
                zip(mn_indices, range(mn_indices.shape[0])):
                
                for c_mn_idx, c_cov_idx in \
                    zip(mn_indices, range(mn_indices.shape[0])):
                    
                    Q[r_mn_idx, c_mn_idx, idx] = tcov[r_cov_idx, c_cov_idx, :]

        # Compute the probability mass in the truncated region
        alpha[idx] = talpha.reshape([-1])
        
    return x_hat, Q, alpha

def perform_truncated_em(x, K, truncated_bounds, censoring_bounds, 
                         pp, mu, sigma, max_iteration=200):
    """Estimate GMM's parameters by using EM algorithm for truncated and
       censored data.
    
    Args:
        x (2D numpy array): The observed data.
        K (int): The number of mixture component.
        truncated_bounds (2D numpy array): The truncated bounds.
        censoring_bounds (2D numpy array): The censoring bounds. 
        pp (1D numpy array): The initialized component probabilities.
        mu (2D numpy array): The initialized means.
        sigma (3D numpy array): The initialized covariance.
        max_iteration (int): The maximum number of iterations.
        
    Returns:
        results (dictionary): The dictionary containing estimated parameters
            and the logs of the experiment.
    """
    # Get the shapes of the data
    N, D = x.shape
    K = mu.shape[0]

    # Find censoring patterns of the data
    pattern = find_censoring_pattern(x, censoring_bounds)

    # Initialize the array used to track the log likelihood
    ll_old = -np.inf
    ll_hist = np.zeros([max_iteration])

    for it in range(max_iteration):
        # E-STEP

        # Compute the posterior z|x and its log-likelihood
        post, ll = estimate_truncated_censored_loglikelihood(
            x, pp, mu, sigma, pattern, truncated_bounds)

        # Store to the list of history and check convergence
        lldiff = ll - ll_old
        ll_hist[it] = ll
        if (lldiff >= 0) and (lldiff < 10e-7):
            break
        ll_old = ll

        # Compute the unnormalzied truncated component probability
        sum_post = np.sum(post, axis=0)

        # M-STEP
        
        # Consider each component independently
        for k in range(K):
            # 0. Compute the sufficient statistics
            x_hat_e, Q_e, alpha = estimate_sufficient_statistics(
                x, mu[k, :], sigma[:, :, k], pattern)

            # Avoid numerical issue regarding alpha
            alpha0 = (alpha == 0)
            x_hat = x_hat_e
            x_hat[alpha0, :] = 0
            Q = Q_e
            Q[:, :, alpha0] = 0

            # 1. Update mean
            # Estimate the correction term (mk) of the mean
            tc_pdf = norm.pdf(truncated_bounds[0], mu[k, 0],
                              np.sqrt(float(sigma[0, 0, k])))
            mk = tc_pdf / (1 - tc_pdf)

            mu[k, :] = (post[:,k].T.dot(x_hat)) /\
                sum_post[k] - sigma[0, :, k].dot(mk)

            # 2. Update covariance
            # Estimate the correction term (Rk) of the covariance
            Rk = (truncated_bounds[0] - mu[k, 0]) *\
                norm.pdf(truncated_bounds[0], mu[k, 0],\
                np.sqrt(float(sigma[0, 0, k]))) /\
                sigma[0, 0, k] / (1 - norm.cdf(truncated_bounds[0], mu[k, 0],\
                np.sqrt(float(sigma[0, 0, k]))))

            x_hat_0 = (x_hat - mu[k, :]) * np.sqrt(post[:, k].reshape([-1, 1]))

            sigma_new = x_hat_0.T.dot(x_hat_0) + \
                Q.reshape([D*D, N]).dot(post[:, k]).reshape([D, D])
            sigma_new = sigma_new / sum_post[k] - \
                sigma[0, :, k].reshape([1, 2]).T.dot(sigma[0, :, k].\
                reshape([1, 2])).dot(Rk)

            sigma[:, :, k] = (sigma_new + sigma_new.T) / 2

        # Estimate the component probabilities
        sum_post_ratio = np.zeros([K])
        for k in range(K):
            sum_post_ratio[k] = 1 - norm.cdf(truncated_bounds[0],
                                             mu[k, 0], 
                                             np.sqrt(float(sigma[0, 0, k])))
        pp = sum_post / sum_post_ratio
        pp = pp / np.sum(pp)
        
        if it % 10 == 0:
            print("\tIteration #{} \t ** Log-likelihood: {}".format(it, ll))

    # Return the results
    results = {
        'K': K,
        'pp': pp,
        'mu': mu,
        'sigma': sigma,
        'iters': it+1,
        'log_lh': ll,
        'll_hist': ll_hist[:it+1]
    }
    
    return results
