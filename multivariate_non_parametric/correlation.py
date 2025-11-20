import numpy as np
from scipy.stats import rankdata, pearsonr

def partial_spearman(x, y, covariates):
    """
    Compute partial Spearman correlation between x and y,
    controlling for covariates.

    Parameters
    ----------
    x : array-like, shape (n,)
        First continuous variable
    y : array-like, shape (n,)
        Second continuous variable
    covariates : array-like, shape (n, k)
        Covariate(s) to adjust for

    Returns
    -------
    rho : float
        Partial Spearman correlation coefficient
    pval : float
        Two-tailed p-value
    """
    x = np.asarray(x)
    y = np.asarray(y)
    cov = np.asarray(covariates)

    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)

    n = len(x)
    if len(y) != n or cov.shape[0] != n:
        raise ValueError("x, y, and covariates must have the same number of samples.")

    # Step 1: rank-transform
    xr = rankdata(x)
    yr = rankdata(y)
    covr = np.apply_along_axis(rankdata, 0, cov)

    # Step 2: regress ranks on covariates
    Xc = np.column_stack([np.ones(n), covr])  # add intercept

    beta_x, _, _, _ = np.linalg.lstsq(Xc, xr, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(Xc, yr, rcond=None)

    rx = xr - Xc @ beta_x
    ry = yr - Xc @ beta_y

    # Step 3: Pearson correlation of residuals
    rho, pval = pearsonr(rx, ry)

    return rho, pval





######################################################################
####################### DON'T TRY THIS AT HOME #######################
######################################################################

# import numpy as np

# np.random.seed(0)

# n = 50
# x = np.random.normal(size=n)
# y = 0.5 * x + 0.3 * np.random.normal(size=n)
# age = np.random.randint(20, 70, size=n)
# sex = np.random.choice([0, 1], size=n)

# rho, p = partial_spearman(x, y, covariates=np.column_stack([age, sex]))
# print(f"Partial Spearman rho = {rho:.4f}, p = {p:.4g}")