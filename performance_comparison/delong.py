from typing import Tuple
import numpy as np
from scipy import stats


def _compute_pairwise_psi(pos_scores: np.ndarray, neg_scores: np.ndarray) -> np.ndarray:
    """Compute the matrix of indicator values psi(i,j):
    1 if pos_i > neg_j
    0.5 if pos_i == neg_j
    0 if pos_i < neg_j

    Returns shape (n_pos, n_neg).
    """
    # Broadcasting compare
    greater = pos_scores[:, None] > neg_scores[None, :]
    equal = pos_scores[:, None] == neg_scores[None, :]
    psi = greater.astype(float) + 0.5 * equal.astype(float)
    return psi


def delong_auc_variance(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute AUC and DeLong variance estimate for a single set of scores.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground truth labels (1 for positive, 0 for negative).
    y_scores : array-like of shape (n_samples,)
        Target scores. Higher means more likely to be positive.

    Returns
    -------
    auc : float
        Estimated AUC.
    var : float
        Estimated variance of AUC using DeLong's method.
    v01 : ndarray
        Per-positive-row U-statistic contributions (length n_pos).
    v10 : ndarray
        Per-negative-column U-statistic contributions (length n_neg).

    Notes
    -----
    This implementation follows the U-statistics formulation of DeLong: AUC is the
    average of pairwise comparisons between positives and negatives. The variance
    is computed from the variance of the per-observation contributions.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("y_true and y_scores must have the same length")

    # split
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    n_pos = pos_scores.shape[0]
    n_neg = neg_scores.shape[0]
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative samples")

    psi = _compute_pairwise_psi(pos_scores, neg_scores)  # shape (n_pos, n_neg)

    # v01_i = average over negatives of psi(i, j)
    v01 = np.mean(psi, axis=1)
    # v10_j = average over positives of psi(i, j)
    v10 = np.mean(psi, axis=0)

    auc = np.mean(v01)

    if n_pos > 1:
        var_pos = np.var(v01, ddof=1)
    else:
        var_pos = 0.0
    if n_neg > 1:
        var_neg = np.var(v10, ddof=1)
    else:
        var_neg = 0.0

    var_auc = var_pos / n_pos + var_neg / n_neg

    return float(auc), float(var_auc), v01, v10


def delong_roc_test(y_true: np.ndarray, y_scores1: np.ndarray, y_scores2: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
    """Compute DeLong test for two correlated ROC AUCs on the same set of samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground truth labels (1 for positive, 0 for negative).
    y_scores1 : array-like of shape (n_samples,)
        Scores for classifier 1.
    y_scores2 : array-like of shape (n_samples,)
        Scores for classifier 2.

    Returns
    -------
    z : float
        Z statistic for the test (difference in AUCs divided by estimated std error).
    pvalue : float
        Two-sided p-value.
    (auc1, auc2) : tuple of floats
        AUC estimates for classifier 1 and 2.

    Notes
    -----
    The covariance between AUC estimates is computed using the paired per-observation
    U-statistic contributions. The variance of the difference is

        var(d) = var1 + var2 - 2 * cov

    and z = (auc1 - auc2) / sqrt(var(d)).
    """
    y_true = np.asarray(y_true)
    y_scores1 = np.asarray(y_scores1)
    y_scores2 = np.asarray(y_scores2)

    if not (y_true.shape[0] == y_scores1.shape[0] == y_scores2.shape[0]):
        raise ValueError("All inputs must have the same length")

    # compute AUCs and per-sample U contributions
    auc1, var1, v01_1, v10_1 = delong_auc_variance(y_true, y_scores1)
    auc2, var2, v01_2, v10_2 = delong_auc_variance(y_true, y_scores2)

    # covariance terms: cov(v01_1, v01_2)/n_pos + cov(v10_1, v10_2)/n_neg
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)

    # compute covariances with ddof=1 unbiased estimator; if only one sample set var=0
    if n_pos > 1:
        cov_pos = np.cov(v01_1, v01_2, ddof=1)[0, 1]
    else:
        cov_pos = 0.0
    if n_neg > 1:
        cov_neg = np.cov(v10_1, v10_2, ddof=1)[0, 1]
    else:
        cov_neg = 0.0

    cov = cov_pos / n_pos + cov_neg / n_neg

    var_diff = var1 + var2 - 2.0 * cov
    if var_diff <= 0:
        # numerical guard: variance should be non-negative, but can be tiny negative
        if np.isclose(var_diff, 0, atol=1e-12):
            var_diff = 0.0
        else:
            # If negative beyond tolerance, raise error
            raise RuntimeError(f"Calculated negative variance for AUC difference: {var_diff}")

    std_diff = np.sqrt(var_diff)
    z = (auc1 - auc2) / std_diff if std_diff > 0 else 0.0
    pvalue = 2 * stats.norm.sf(abs(z))

    return float(z), float(pvalue), (float(auc1), float(auc2))




"""
Example
-------
>>> import numpy as np
>>> from delong_test import delong_roc_test
>>> y = np.array([1,1,0,0,1,0,1,0])
>>> s1 = np.array([0.9,0.8,0.1,0.2,0.75,0.3,0.6,0.4])
>>> s2 = np.array([0.85,0.7,0.15,0.25,0.6,0.35,0.65,0.45])
>>> z, pvalue, (auc1, auc2) = delong_roc_test(y, s1, s2)
>>> print(auc1, auc2, z, pvalue)

"""
