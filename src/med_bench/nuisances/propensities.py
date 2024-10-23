"""
the objective of this script is to provide nuisance estimators 
for mediation in causal inference
"""

import numpy as np


from med_bench.utils.utils import _get_train_test_lists


def estimate_treatment_propensity_x(t, m, x, crossfit, clf_t_x):
    """
    Estimate treatment probabilities P(T=1|X) with train
    test lists from crossfitting

    Returns
    -------
    p_x : array-like, shape (n_samples)
        probabilities P(T=1|X)
    p_xm : array-like, shape (n_samples)
        probabilities P(T=1|X, M)
    """
    n = len(t)

    p_x, p_xm = [np.zeros(n) for h in range(2)]
    # compute propensity scores
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)
    if len(t.shape) == 1:
        t = t.reshape(-1, 1)

    train_test_list = _get_train_test_lists(crossfit, n, x)

    for train_index, test_index in train_test_list:

        # predict P(T=1|X), P(T=1|X, M)
        p_x[test_index] = clf_t_x.predict_proba(x[test_index, :])[:, 1]

    return p_x

def estimate_treatment_probabilities(t, m, x, crossfit, clf_t_x, clf_t_xm, fit=False):
    """
    Estimate treatment probabilities P(T=1|X) and P(T=1|X, M) with train
    test lists from crossfitting

    Returns
    -------
    p_x : array-like, shape (n_samples)
        probabilities P(T=1|X)
    p_xm : array-like, shape (n_samples)
        probabilities P(T=1|X, M)
    """
    n = len(t)

    p_x, p_xm = [np.zeros(n) for h in range(2)]
    # compute propensity scores
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)
    if len(t.shape) == 1:
        t = t.reshape(-1, 1)

    train_test_list = _get_train_test_lists(crossfit, n, x)

    xm = np.hstack((x, m))

    for train_index, test_index in train_test_list:

        # predict P(T=1|X), P(T=1|X, M)
        p_x[test_index] = clf_t_x.predict_proba(x[test_index, :])[:, 1]
        p_xm[test_index] = clf_t_xm.predict_proba(xm[test_index, :])[:, 1]

    return p_x, p_xm