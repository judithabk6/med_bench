"""
the objective of this script is to provide nuisance estimators 
for mediation in causal inference
"""

import numpy as np

from med_bench.utils.utils import _get_train_test_lists, _get_interactions


def estimate_conditional_mean_outcome(
    t, m, x, y, crossfit, reg_y, interaction, fit=False
):
    """
    Estimate conditional mean outcome E[Y|T,M,X]
    with train test lists from crossfitting

    Returns
    -------
    mu_t0: list
        contains array-like, shape (n_samples) conditional mean outcome estimates E[Y|T=0,M=m,X]
    mu_t1, list
        contains array-like, shape (n_samples) conditional mean outcome estimates E[Y|T=1,M=m,X]
    mu_m0x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=0,M,X]
    mu_m1x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=1,M,X]
    """
    n = len(y)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        mr = m.reshape(-1, 1)
    else:
        mr = np.copy(m)
    if len(t.shape) == 1:
        t = t.reshape(-1, 1)

    t0 = np.zeros((n, 1))
    t1 = np.ones((n, 1))
    m1 = np.ones((n, 1))

    train_test_list = _get_train_test_lists(crossfit, n, x)

    mu_1mx, mu_0mx = [np.zeros(n) for _ in range(2)]
    mu_t1, mu_t0 = [], []

    m1 = np.ones((n, 1))

    x_t_mr = _get_interactions(interaction, x, t, mr)

    x_t1_m = _get_interactions(interaction, x, t1, m)
    x_t0_m = _get_interactions(interaction, x, t0, m)

    for train_index, test_index in train_test_list:

        # mu_tm model fitting
        if fit == True:
            reg_y = reg_y.fit(x_t_mr[train_index, :], y[train_index])

        # predict E[Y|T=t,M,X]
        mu_0mx[test_index] = reg_y.predict(x_t0_m[test_index, :]).squeeze()
        mu_1mx[test_index] = reg_y.predict(x_t1_m[test_index, :]).squeeze()

        for i, b in enumerate(np.unique(m)):
            mu_1bx, mu_0bx = [np.zeros(n) for h in range(2)]
            mb = m1 * b

            # predict E[Y|T=t,M=m,X]
            mu_0bx[test_index] = reg_y.predict(
                _get_interactions(interaction, x, t0, mb)[test_index, :]
            ).squeeze()
            mu_1bx[test_index] = reg_y.predict(
                _get_interactions(interaction, x, t1, mb)[test_index, :]
            ).squeeze()

            mu_t0.append(mu_0bx)
            mu_t1.append(mu_1bx)

    return mu_t0, mu_t1, mu_0mx, mu_1mx
