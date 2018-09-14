__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import sys
import warnings

import numpy as np

from experiments import perform_bivariate_3_gaussians_exp

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    

def main(argv):
    # Number of data points
    N = 100000

    # Define the true values of mixture weights
    pp = np.array([0.5, 0.2, 0.3])

    # Define Gaussian components
    mu_1 = np.array([-3., 3.])
    mu_2 = np.array([10., -1.])
    mu_3 = np.array([20., 20.])
    sigma_1 = np.diag([20., 5.])
    sigma_2 = np.diag([5., 20.])
    sigma_3 = np.diag([20., 20.])

    # Define the truncation and censoring bounds
    truncation_bounds = np.array([0, np.inf])
    censoring_bounds = np.array([[0], [25]])

    # Start the experiment 
    perform_bivariate_3_gaussians_exp(N, pp, mu_1, mu_2, mu_3,
                                      sigma_1, sigma_2, sigma_3,
                                      truncation_bounds, censoring_bounds)

if __name__ == "__main__":
    main(sys.argv)
