"""
the objective of this script is to implement nuisances functions
used in mediation estimators in causal inference
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy import stats
from scipy.special import expit
from scipy.stats import bernoulli
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from .utils import _convert_array_to_R, _get_interactions

ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5
TINY = 1.e-12


def _get_train_test_lists(crossfit, n, x):
    """
    Obtain train and test folds

    Returns
    -------
    train_test_list : list
        indexes with train and test indexes
    """
    if crossfit < 2:
        train_test_list = [[np.arange(n), np.arange(n)]]
    else:
        kf = KFold(n_splits=crossfit)
        train_test_list = list()
        for train_index, test_index in kf.split(x):
            train_test_list.append([train_index, test_index])
    return train_test_list


def _get_regularization_parameters(regularization):
    """
    Obtain regularization parameters

    Returns
    -------
    cs : list
        each of the values in Cs describes the inverse of regularization
        strength for predictors
    alphas : list
        alpha values to try in ridge models
    """
    if regularization:
        alphas = ALPHAS
        cs = ALPHAS
    else:
        alphas = [TINY]
        cs = [np.inf]

    return cs, alphas


def _get_classifier(regularization, forest, calibration, random_state=42):
    """
    Obtain context classifiers to estimate treatment probabilities.

    Returns
    -------
    clf : classifier on contexts, etc. for predicting P(T=1|X),
          P(T=1|X, M) or f(M|T,X)
    """
    cs, _ = _get_regularization_parameters(regularization)

    if not forest:
        clf = LogisticRegressionCV(random_state=random_state, Cs=cs,
                                   cv=CV_FOLDS)
    else:
        clf = RandomForestClassifier(random_state=random_state,
                                     n_estimators=100, min_samples_leaf=10)
    if calibration in {"sigmoid", "isotonic"}:
        clf = CalibratedClassifierCV(clf, method=calibration)

    return clf


def _get_regressor(regularization, forest, random_state=42):
    """
    Obtain regressors to estimate conditional mean outcomes.

    Returns
    -------
    reg : regressor on contexts, etc. for predicting E[Y|T,M,X], etc.
    """
    _, alphas = _get_regularization_parameters(regularization)

    if not forest:
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
    else:
        reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)

    return reg


def _estimate_treatment_probabilities(t, m, x, crossfit, clf_t_x, clf_t_xm):
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

        # p_x, p_xm model fitting
        clf_t_x = clf_t_x.fit(x[train_index, :], t[train_index])
        clf_t_xm = clf_t_xm.fit(xm[train_index, :], t[train_index])

        # predict P(T=1|X), P(T=1|X, M)
        p_x[test_index] = clf_t_x.predict_proba(x[test_index, :])[:, 1]
        p_xm[test_index] = clf_t_xm.predict_proba(xm[test_index, :])[:, 1]

    return p_x, p_xm


def _estimate_mediator_density(t, m, x, y, crossfit, clf_m, interaction):
    """
    Estimate mediator density f(M|T,X)
    with train test lists from crossfitting

    Returns
    -------
    f_00x: array-like, shape (n_samples)
        probabilities f(M=0|T=0,X)
    f_01x, array-like, shape (n_samples)
        probabilities f(M=0|T=1,X)
    f_10x, array-like, shape (n_samples)
        probabilities f(M=1|T=0,X)
    f_11x, array-like, shape (n_samples)
        probabilities f(M=1|T=1,X)
    f_m0x, array-like, shape (n_samples)
        probabilities f(M|T=0,X)
    f_m1x, array-like, shape (n_samples)
        probabilities f(M|T=1,X)
    """
    n = len(y)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(t.shape) == 1:
        t = t.reshape(-1, 1)

    t0 = np.zeros((n, 1))
    t1 = np.ones((n, 1))

    m = m.ravel()

    train_test_list = _get_train_test_lists(crossfit, n, x)

    f_00x, f_01x, f_10x, f_11x, f_m0x, f_m1x = [np.zeros(n) for _ in range(6)]

    t_x = _get_interactions(interaction, t, x)

    t0_x = _get_interactions(interaction, t0, x)
    t1_x = _get_interactions(interaction, t1, x)

    for train_index, test_index in train_test_list:

        test_ind = np.arange(len(test_index))

        # f_mtx model fitting
        clf_m = clf_m.fit(t_x[train_index, :], m[train_index])
        #clf_m = clf_m.fit(t_x[train_index, :], m.ravel()[train_index])

        # predict f(M=m|T=t,X)
        fm_0 = clf_m.predict_proba(t0_x[test_index, :])
        f_00x[test_index] = fm_0[:, 0]
        f_01x[test_index] = fm_0[:, 1]
        fm_1 = clf_m.predict_proba(t1_x[test_index, :])
        f_10x[test_index] = fm_1[:, 0]
        f_11x[test_index] = fm_1[:, 1]

        # predict f(M|T=t,X)
        f_m0x[test_index] = fm_0[test_ind, m[test_index].astype(int)]
        f_m1x[test_index] = fm_1[test_ind, m[test_index].astype(int)]

    return f_00x, f_01x, f_10x, f_11x, f_m0x, f_m1x


def _estimate_conditional_mean_outcome(t, m, x, y, crossfit, reg_y,
                                       interaction):
    """
    Estimate conditional mean outcome E[Y|T,M,X]
    with train test lists from crossfitting

    Returns
    -------
    mu_00x: array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=0,M=0,X]
    mu_01x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=0,M=1,X]
    mu_10x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=1,M=0,X]
    mu_11x, array-like, shape (n_samples)
        conditional mean outcome estimates E[Y|T=1,M=1,X]
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
    m0 = np.zeros((n, 1))
    m1 = np.ones((n, 1))

    train_test_list = _get_train_test_lists(crossfit, n, x)

    mu_11x, mu_10x, mu_01x, mu_00x, mu_1mx, mu_0mx = [np.zeros(n) for _ in
                                                      range(6)]

    x_t_mr = _get_interactions(interaction, x, t, mr)

    x_t1_m1 = _get_interactions(interaction, x, t1, m1)
    x_t1_m0 = _get_interactions(interaction, x, t1, m0)
    x_t0_m1 = _get_interactions(interaction, x, t0, m1)
    x_t0_m0 = _get_interactions(interaction, x, t0, m0)

    x_t1_m = _get_interactions(interaction, x, t1, m)
    x_t0_m = _get_interactions(interaction, x, t0, m)

    for train_index, test_index in train_test_list:

        # mu_tm model fitting
        reg_y = reg_y.fit(x_t_mr[train_index, :], y[train_index])

        # predict E[Y|T=t,M=m,X]
        mu_00x[test_index] = reg_y.predict(x_t0_m0[test_index, :])
        mu_01x[test_index] = reg_y.predict(x_t0_m1[test_index, :])
        mu_10x[test_index] = reg_y.predict(x_t1_m0[test_index, :])
        mu_11x[test_index] = reg_y.predict(x_t1_m1[test_index, :])

        # predict E[Y|T=t,M,X]
        mu_0mx[test_index] = reg_y.predict(x_t0_m[test_index, :])
        mu_1mx[test_index] = reg_y.predict(x_t1_m[test_index, :])

    return mu_00x, mu_01x, mu_10x, mu_11x, mu_0mx, mu_1mx


def _estimate_cross_conditional_mean_outcome(t, m, x, y, crossfit, reg_y,
                                             reg_cross_y, f, interaction):
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
        mu_11x,  # E[Y|T=1,M=1,X]
        mu_10x,  # E[Y|T=1,M=0,X]
        mu_01x,  # E[Y|T=0,M=1,X]
        mu_00x,  # E[Y|T=0,M=0,X]
        E_mu_t0_t0,  # E[E[Y|T=0,M,X]|T=0,X]
        E_mu_t0_t1,  # E[E[Y|T=0,M,X]|T=1,X]
        E_mu_t1_t0,  # E[E[Y|T=1,M,X]|T=0,X]
        E_mu_t1_t1,  # E[E[Y|T=1,M,X]|T=1,X]
    ) = [np.zeros(n) for _ in range(10)]

    t0, m0 = np.zeros((n, 1)), np.zeros((n, 1))
    t1, m1 = np.ones((n, 1)), np.ones((n, 1))

    train_test_list = _get_train_test_lists(crossfit, n, x)

    x_t_m = _get_interactions(interaction, x, t, m)
    x_t1_m = _get_interactions(interaction, x, t1, m)
    x_t0_m = _get_interactions(interaction, x, t0, m)

    x_t0_m0 = _get_interactions(interaction, x, t0, m0)
    x_t0_m1 = _get_interactions(interaction, x, t0, m1)
    x_t1_m0 = _get_interactions(interaction, x, t1, m0)
    x_t1_m1 = _get_interactions(interaction, x, t1, m1)

    f_00x, f_01x, f_10x, f_11x = f

    # Cross-fitting loop
    for train_index, test_index in train_test_list:
        # Index declaration
        ind_t0 = t[test_index] == 0

        # mu_tm model fitting
        reg_y = reg_y.fit(x_t_m[train_index, :], y[train_index])

        # predict E[Y|T=t,M,X]
        mu_1mx[test_index] = reg_y.predict(x_t1_m[test_index, :])
        mu_0mx[test_index] = reg_y.predict(x_t0_m[test_index, :])

        # predict E[Y|T=t,M=m,X]
        mu_00x[test_index] = reg_y.predict(x_t0_m0[test_index, :])
        mu_01x[test_index] = reg_y.predict(x_t0_m1[test_index, :])
        mu_11x[test_index] = reg_y.predict(x_t1_m1[test_index, :])
        mu_10x[test_index] = reg_y.predict(x_t1_m0[test_index, :])

        # E[E[Y|T=1,M=m,X]|T=t,X] model fitting
        reg_y_t1m1_t0 = clone(reg_cross_y).fit(
            x[test_index, :][ind_t0, :], mu_11x[test_index][ind_t0]
        )
        reg_y_t1m0_t0 = clone(reg_cross_y).fit(
            x[test_index, :][ind_t0, :], mu_10x[test_index][ind_t0]
        )
        reg_y_t1m1_t1 = clone(reg_cross_y).fit(
            x[test_index, :][~ind_t0, :], mu_11x[test_index][~ind_t0]
        )
        reg_y_t1m0_t1 = clone(reg_cross_y).fit(
            x[test_index, :][~ind_t0, :], mu_10x[test_index][~ind_t0]
        )

        # predict E[E[Y|T=1,M=m,X]|T=t,X]
        E_mu_t1_t0[test_index] = (
                reg_y_t1m0_t0.predict(x[test_index, :]) * f_00x[test_index]
                + reg_y_t1m1_t0.predict(x[test_index, :]) * f_01x[test_index]
        )
        E_mu_t1_t1[test_index] = (
                reg_y_t1m0_t1.predict(x[test_index, :]) * f_10x[test_index]
                + reg_y_t1m1_t1.predict(x[test_index, :]) * f_11x[test_index]
        )

        # E[E[Y|T=0,M=m,X]|T=t,X] model fitting
        reg_y_t0m1_t0 = clone(reg_cross_y).fit(
            x[test_index, :][ind_t0, :], mu_01x[test_index][ind_t0]
        )
        reg_y_t0m0_t0 = clone(reg_cross_y).fit(
            x[test_index, :][ind_t0, :], mu_00x[test_index][ind_t0]
        )
        reg_y_t0m1_t1 = clone(reg_cross_y).fit(
            x[test_index, :][~ind_t0, :], mu_01x[test_index][~ind_t0]
        )
        reg_y_t0m0_t1 = clone(reg_cross_y).fit(
            x[test_index, :][~ind_t0, :], mu_00x[test_index][~ind_t0]
        )

        # predict E[E[Y|T=0,M=m,X]|T=t,X]
        E_mu_t0_t0[test_index] = (
                reg_y_t0m0_t0.predict(x[test_index, :]) * f_00x[test_index]
                + reg_y_t0m1_t0.predict(x[test_index, :]) * f_01x[test_index]
        )
        E_mu_t0_t1[test_index] = (
                reg_y_t0m0_t1.predict(x[test_index, :]) * f_10x[test_index]
                + reg_y_t0m1_t1.predict(x[test_index, :]) * f_11x[test_index]
        )

    return mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1


def _estimate_cross_conditional_mean_outcome_nesting(t, m, x, y, crossfit,
                                                     reg_y, reg_cross_y):
    """
    Estimate treatment probabilities and the conditional mean outcome,
    cross conditional mean outcome

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

    # initialisation
    (
        mu_1mx,  # E[Y|T=1,M,X]
        mu_1mx_nested,  # E[Y|T=1,M,X] predicted on train_nested set
        mu_0mx,  # E[Y|T=0,M,X]
        mu_0mx_nested,  # E[Y|T=0,M,X] predicted on train_nested set
        E_mu_t1_t0,  # E[E[Y|T=1,M,X]|T=0,X]
        E_mu_t0_t1,  # E[E[Y|T=0,M,X]|T=1,X]
        mu_1x,  # E[Y|T=1,X]
        mu_0x,  # E[Y|T=0,X]
    ) = [np.zeros(n) for _ in range(8)]

    xm = np.hstack((x, m))

    train_test_list = _get_train_test_lists(crossfit, n, x)

    for train, test in train_test_list:
        # define test set
        train1 = train[t[train] == 1]
        train0 = train[t[train] == 0]

        train_mean, train_nested = np.array_split(train, 2)
        train_mean1 = train_mean[t[train_mean] == 1]
        train_mean0 = train_mean[t[train_mean] == 0]
        train_nested1 = train_nested[t[train_nested] == 1]
        train_nested0 = train_nested[t[train_nested] == 0]

        # predict E[Y|T=1,M,X]
        reg_y1m = clone(reg_y)
        reg_y1m.fit(xm[train_mean1], y[train_mean1])
        mu_1mx[test] = reg_y1m.predict(xm[test])
        mu_1mx_nested[train_nested] = reg_y1m.predict(xm[train_nested])

        # predict E[Y|T=0,M,X]
        reg_y0m = clone(reg_y)
        reg_y0m.fit(xm[train_mean0], y[train_mean0])
        mu_0mx[test] = reg_y0m.predict(xm[test])
        mu_0mx_nested[train_nested] = reg_y0m.predict(xm[train_nested])

        # predict E[E[Y|T=1,M,X]|T=0,X]
        reg_cross_y1 = clone(reg_cross_y)
        reg_cross_y1.fit(x[train_nested0], mu_1mx_nested[train_nested0])
        E_mu_t1_t0[test] = reg_cross_y1.predict(x[test])

        # predict E[E[Y|T=0,M,X]|T=1,X]
        reg_cross_y0 = clone(reg_cross_y)
        reg_cross_y0.fit(x[train_nested1], mu_0mx_nested[train_nested1])
        E_mu_t0_t1[test] = reg_cross_y0.predict(x[test])

        # predict E[Y|T=1,X]
        reg_y1 = clone(reg_y)
        reg_y1.fit(x[train1], y[train1])
        mu_1x[test] = reg_y1.predict(x[test])

        # predict E[Y|T=0,X]
        reg_y0 = clone(reg_y)
        reg_y0.fit(x[train0], y[train0])
        mu_0x[test] = reg_y0.predict(x[test])

    return mu_0mx, mu_1mx, mu_0x, E_mu_t0_t1, E_mu_t1_t0, mu_1x
