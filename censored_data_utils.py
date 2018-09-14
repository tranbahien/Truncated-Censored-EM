"""censored_data_utils.py: Utilities used to censor and truncate datasets."""

__author__      = "Ba-Hien TRAN"
__email__       = "bahientranvn@gmail.com"


import numpy as np


def censor_and_truncate_data(y, c_up=25, c_low=0, t_up=np.inf, t_low=0):
    """Perform censoring and truncation on the data.
    
    Args:
        y (2D numpy array): The original data.
        c_up (float): The upper censoring bound.
        c_low (float): The lower censoring bound.
        t_up (float): The upper truncation bound.
        t_low (float): The lower truncation bound.
        
    Returns:
        x (2D numpy array): The censored and truncated data.
    """
    # Truncation index (only truncate the first dimension)
    idxt = y[:, 0] > t_low

    # Censoring index
    idxcu = y > c_up
    idxcl = y < c_low

    # Perform censoring
    x = y
    x[idxcu] = c_up
    x[idxcl] = c_low

    # Then perform truncation
    x = x[idxt, :]

    return x

def find_censoring_pattern(x, bounds):
    """Find the censoring pattern from the truncated and censored data.
    
    Args:
        x (2D numpy array): The data points.
        bounds (2D numpy array): The lower and upper censoring bounds.
        
    Returns:
        pattern (dictionary): The desired pattern.
    """
    # Get the number of data points and the number of dimensions
    # of each data point
    N, D = x.shape

    # Get the lower and upper censoring bounds
    if bounds.shape[0] == 1:
        bounds = bounds.T
    if bounds.shape[1] < D:
        bounds = bounds * np.ones([1, D])
    lower_bounds = bounds[0, :]
    upper_bounds = bounds[1, :]

    # Find the censored coordiantes
    censored = (x > lower_bounds) & (x < upper_bounds)

    # Find the datapoints that are completely observed
    complete_instances = censored[:, 0] & censored[:, 1]

    # Estimate the integral region for each datapoint
    integral_regions = []
    for n in range(N):
        # Get the uncensored locations
        uncens_loc = ~censored[n, :]
        
        # Check if the datpoint is fully observed
        if uncens_loc.sum() > 0:
            reg_lower_bound = upper_bounds[uncens_loc]
            reg_upper_bound = lower_bounds[uncens_loc]
            
            reg_lower_bound[x[n, uncens_loc] == lower_bounds[uncens_loc]] =\
                np.array(-np.inf)
            reg_upper_bound[x[n, uncens_loc] == upper_bounds[uncens_loc]] =\
                np.array(np.inf)
            
            integral_regions.append([reg_lower_bound, reg_upper_bound])
        else:
            integral_regions.append([[], []])
            
    # Re-order the censoring pattern
    unique_pattern = np.unique(censored, axis=0)
    same_pattern = np.zeros([N, unique_pattern.shape[0]], dtype=np.bool)

    for k in range(unique_pattern.shape[0]):
        same_pattern[:, k] = (censored == unique_pattern[k, :]).all(axis=1)

    num_patterns = sum(same_pattern)

    idx = num_patterns.argsort()[::-1]
    num_patterns = num_patterns[idx]

    unique_pattern = unique_pattern[idx, :]
    same_pattern = same_pattern[:, idx]

    # Return the dictionary containing the censoring patterns
    pattern = {
        'censored': censored,
        'complete': complete_instances,
        'range' : integral_regions,
        'uniq_censored': unique_pattern,
        'count': num_patterns,
        'xpttn': same_pattern
    }

    return pattern
