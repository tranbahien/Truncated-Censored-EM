"""gmm.py: Utilities used to generate and evaluate GMM data."""

__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import numpy as np

from sklearn import mixture


def generate_gmm_data(pi, mu, sigma, N):
    """Generate synthetic data based on Gaussian Mixture Model (GMM).
    
    Args:
        pi (1D numpy array): The mixing weights.
        mu (2D numpy array): The means of Gaussian components.
        sigma (3D numpy array): The covariance matrices of Gaussian components.
        N (int): The number of generated data points.
        
    Returns:
        x (2D numpy array): The generated data points.
    """
    # Get the number of components
    K = pi.shape[0]

    # Get the number of dimensions
    D = mu.shape[1]

    # Get the component weights
    pi = np.concatenate(([0], np.cumsum(pi[:-1])))

    # Generate the latent variables
    z = np.sum(np.random.uniform(size=[N, 1]) > np.tile(pi, (N, 1)), axis=1) - 1
    
    # Generate dataset
    x = np.zeros([N, D])

    for k in range(K):
        x[z == k, :] = np.random.multivariate_normal(mu[k, :], 
                                                     sigma[:, :, k], 
                                                     np.sum(z==k))
        
    return x


def estimate_kl_divergence_gmm(gmm_p, gmm_q):
    """Compute KL-divergence betweent two Gaussian Mixture Models.
    
    Args:
        gmm_p ((mixture.GaussianMixture object)): The GMM model corresponding
            to the true parameters.
        gmm_q ((mixture.GaussianMixture object)): The GMM model corresponding
            to the estimated parameters.

    Returns:
        kl_divergence (float): The estimated KL-divergence.
    """
    X, _ = gmm_p.sample(1e7)
    log_p_X = gmm_p.score(X)
    log_q_X = gmm_q.score(X)

    kl_divergence = log_p_X.mean() - log_q_X.mean()

    return kl_divergence

def build_GMM_model(pp, mu, sigma, seed):
    """Build a Gaussian Mixture Model based on scikit-learn.

    Args:
        pp (1D numpy array): The mixing weights of GMM.
        mu (2D numpy array): The means of components of GMM.
        sigma (3D numpy array): The covariances of components of GMM.
        seed (int): The random seed for reproducibility.

    Returns:
        model (mixture.GaussianMixture object): The built GMM model.
    """
    K, D = mu.shape
    model = mixture.GaussianMixture(n_components=K, random_state=seed)
    model.fit(np.random.rand(10, D))
    model.weights_ = pp
    model.means_ = mu
    model.covariances_ = sigma.T

    return model

def reorder_gmm_compoments(pp, mu, sigma):
    """Re-order GMM components based on their mixing weights.

    Args:
        pp (1D numpy array): The mixing weights of GMM.
        mu (2D numpy array): The means of components of GMM.
        sigma (3D numpy array): The covariances of components of GMM.

    Returns:
        new_pp (1D numpy array): The re-ordered mixing weights of GMM.
        new_mu (2D numpy array): The re-ordered means of components of GMM.
        new_sigma (3D numpy array): The re-ordered covariances of components
            of GMM.
    """
    idx = pp.argsort()
    new_pp = pp[idx]
    new_mu = mu[idx, :]
    new_sigma = sigma.T[idx, :].T

    return new_pp, new_mu, new_sigma
