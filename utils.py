import numpy as np
from scipy.stats import norm, binom, bernoulli
from scipy.stats import pearsonr

def my_fisherz(    
    X,
    Y,
    condition_on = None,
    correlation_matrix = None,
    ):

    if condition_on is None:
        condition_on = np.empty((X.shape[0], 0))

    # compute the correlation matrix within the specified data
    data = np.hstack((X, Y, condition_on))
    sample_size = data.shape[0]
    if correlation_matrix is None:
        correlation_matrix = np.corrcoef(data.T)

    inv = np.linalg.pinv(correlation_matrix)
    r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])

    # apply the Fisher Z-transformation
    Z = 0.5 * np.log((1 + r) / (1 - r))

    # compute the test statistic
    statistic = np.sqrt(sample_size - condition_on.shape[1] - 3) * abs(Z)
    p = 2 * (1 - norm.cdf(abs(statistic)))
    
    print(f'{statistic:.4f}, {p:.4f}')