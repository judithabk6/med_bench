"""
the objective of this script is to provide nuisance estimators 
for mediation in causal inference
"""

import numpy as np

from sklearn.base import clone

from med_bench.utils.utils import _get_train_test_lists, _get_interactions


def estimate_cross_conditional_mean_outcome_discrete(m, x, y, f, regressors):
    """
    Estimate the conditional mean outcome,
    the cross conditional mean outcome

    Returns
    -------
    mu_m0x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=0,M,X]
    mu_m1x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=1,M,X]
    E_mu_t0_t0, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=0,X]
    E_mu_t0_t1, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=1,X]
    E_mu_t1_t0, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=0,X]
    E_mu_t1_t1, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=1,X]
    """
    n = len(y)

    # Initialisation
    (
        mu_1mx,  # E[Y|T=1,M,X]
        mu_0mx,  # E[Y|T=0,M,X]
        E_mu_t0_t0,  # E[E[Y|T=0,M,X]|T=0,X]
        E_mu_t0_t1,  # E[E[Y|T=0,M,X]|T=1,X]
        E_mu_t1_t0,  # E[E[Y|T=1,M,X]|T=0,X]
        E_mu_t1_t1,  # E[E[Y|T=1,M,X]|T=1,X]
    ) = [np.zeros(n) for _ in range(6)]

    t0, m0 = np.zeros((n, 1)), np.zeros((n, 1))
    t1, m1 = np.ones((n, 1)), np.ones((n, 1))

    x_t1_m = _get_interactions(False, x, t1, m)
    x_t0_m = _get_interactions(False, x, t0, m)

    f_t0, f_t1 = f

    # Index declaration
    test_index = np.arange(n)

    # predict E[Y|T=t,M,X]
    mu_1mx[test_index] = regressors["y_t_mx"].predict(x_t1_m[test_index, :])
    mu_0mx[test_index] = regressors["y_t_mx"].predict(x_t0_m[test_index, :])

    for i, b in enumerate(np.unique(m)):

        # f(M=m|T=t,X)
        f_0bx, f_1bx = f_t0[i], f_t1[i]

        # predict E[E[Y|T=1,M=m,X]|T=t,X]
        E_mu_t1_t0[test_index] += (
            regressors["reg_y_t1m{}_t0".format(i)].predict(x[test_index, :])
            * f_0bx[test_index]
        )
        E_mu_t1_t1[test_index] += (
            regressors["reg_y_t1m{}_t1".format(i)].predict(x[test_index, :])
            * f_1bx[test_index]
        )

        # predict E[E[Y|T=0,M=m,X]|T=t,X]
        E_mu_t0_t0[test_index] += (
            regressors["reg_y_t0m{}_t0".format(i)].predict(x[test_index, :])
            * f_0bx[test_index]
        )
        E_mu_t0_t1[test_index] += (
            regressors["reg_y_t0m{}_t1".format(i)].predict(x[test_index, :])
            * f_1bx[test_index]
        )

    return mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1


def estimate_cross_conditional_mean_outcome(
    t, m, x, y, crossfit, reg_y, reg_cross_y, f, interaction
):
    """
    Estimate the conditional mean outcome,
    the cross conditional mean outcome

    Returns
    -------
    mu_m0x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=0,M,X]
    mu_m1x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=1,M,X]
    E_mu_t0_t0, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=0,X]
    E_mu_t0_t1, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=1,X]
    E_mu_t1_t0, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=0,X]
    E_mu_t1_t1, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=1,X]
    """
    n = len(y)

    # Initialisation
    (
        mu_1mx,  # E[Y|T=1,M,X]
        mu_0mx,  # E[Y|T=0,M,X]
        E_mu_t0_t0,  # E[E[Y|T=0,M,X]|T=0,X]
        E_mu_t0_t1,  # E[E[Y|T=0,M,X]|T=1,X]
        E_mu_t1_t0,  # E[E[Y|T=1,M,X]|T=0,X]
        E_mu_t1_t1,  # E[E[Y|T=1,M,X]|T=1,X]
    ) = [np.zeros(n) for _ in range(6)]

    t0, m0 = np.zeros((n, 1)), np.zeros((n, 1))
    t1, m1 = np.ones((n, 1)), np.ones((n, 1))

    train_test_list = _get_train_test_lists(crossfit, n, x)

    x_t_m = _get_interactions(interaction, x, t, m)
    x_t1_m = _get_interactions(interaction, x, t1, m)
    x_t0_m = _get_interactions(interaction, x, t0, m)

    f_t0, f_t1 = f

    # Cross-fitting loop
    for train_index, test_index in train_test_list:
        # Index declaration
        ind_t0 = t[test_index] == 0

        # mu_tm model fitting
        reg_y = reg_y.fit(x_t_m[train_index, :], y[train_index])

        # predict E[Y|T=t,M,X]
        mu_1mx[test_index] = reg_y.predict(x_t1_m[test_index, :])
        mu_0mx[test_index] = reg_y.predict(x_t0_m[test_index, :])

        for i, b in enumerate(np.unique(m)):
            mb = m1 * b

            mu_1bx, mu_0bx, f_0bx, f_1bx = [np.zeros(n) for h in range(4)]

            # f(M=m|T=t,X)
            f_0bx, f_1bx = f_t0[i], f_t1[i]

            # predict E[Y|T=t,M=m,X]
            mu_0bx[test_index] = reg_y.predict(
                _get_interactions(interaction, x, t0, mb)[test_index, :]
            )
            mu_1bx[test_index] = reg_y.predict(
                _get_interactions(interaction, x, t1, mb)[test_index, :]
            )

            # E[E[Y|T=1,M=m,X]|T=t,X] model fitting
            reg_y_t1mb_t0 = clone(reg_cross_y).fit(
                x[test_index, :][ind_t0, :], mu_1bx[test_index][ind_t0]
            )
            reg_y_t1mb_t1 = clone(reg_cross_y).fit(
                x[test_index, :][~ind_t0, :], mu_1bx[test_index][~ind_t0]
            )

            # predict E[E[Y|T=1,M=m,X]|T=t,X]
            E_mu_t1_t0[test_index] += (
                reg_y_t1mb_t0.predict(x[test_index, :]) * f_0bx[test_index]
            )
            E_mu_t1_t1[test_index] += (
                reg_y_t1mb_t1.predict(x[test_index, :]) * f_1bx[test_index]
            )

            # E[E[Y|T=0,M=m,X]|T=t,X] model fitting
            reg_y_t0mb_t0 = clone(reg_cross_y).fit(
                x[test_index, :][ind_t0, :], mu_0bx[test_index][ind_t0]
            )
            reg_y_t0mb_t1 = clone(reg_cross_y).fit(
                x[test_index, :][~ind_t0, :], mu_0bx[test_index][~ind_t0]
            )

            # predict E[E[Y|T=0,M=m,X]|T=t,X]
            E_mu_t0_t0[test_index] += (
                reg_y_t0mb_t0.predict(x[test_index, :]) * f_0bx[test_index]
            )
            E_mu_t0_t1[test_index] += (
                reg_y_t0mb_t1.predict(x[test_index, :]) * f_1bx[test_index]
            )

    return mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1


def estimate_cross_conditional_mean_outcome_nesting(m, x, y, regressors):
    """
    Estimate the conditional mean outcome,
    the cross conditional mean outcome

    Returns
    -------
    mu_m0x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=0,M,X]
    mu_m1x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=1,M,X]
    mu_0x, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=0,X]
    E_mu_t0_t1, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=1,X]
    E_mu_t1_t0, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=0,X]
    mu_1x, array-like, shape (n_samples)
        cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=1,X]
    """
    n = len(y)

    xm = np.hstack((x, m))

    # predict E[Y|T=1,M,X]
    mu_1mx = regressors["y_t1_mx"].predict(xm)

    # predict E[Y|T=0,M,X]
    mu_0mx = regressors["y_t0_mx"].predict(xm)

    # predict E[E[Y|T=1,M,X]|T=0,X]
    E_mu_t1_t0 = regressors["y_t1_x_t0"].predict(x)

    # predict E[E[Y|T=0,M,X]|T=1,X]
    E_mu_t0_t1 = regressors["y_t0_x_t1"].predict(x)

    # predict E[Y|T=1,X]
    mu_1x = regressors["y_t1_x"].predict(x)

    # predict E[Y|T=0,X]
    mu_0x = regressors["y_t0_x"].predict(x)

    return mu_0mx, mu_1mx, mu_0x, E_mu_t0_t1, E_mu_t1_t0, mu_1x
