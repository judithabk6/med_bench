from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from numpy.random import default_rng
from scipy import stats
import pandas as pd
from pathlib import Path
from scipy.stats import bernoulli
from scipy.special import expit

from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.base import clone
from .utils import get_interactions, _convert_array_to_R

ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5
TINY = 1.e-12

def get_x_classifiers(regularization, forest, calibration, calib_method):

    if regularization:
        cs = ALPHAS
    else:
        cs = [np.inf]

    if not forest:
        x_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)
        xm_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)
    else:
        x_clf = RandomForestClassifier(n_estimators=100,
                                       min_samples_leaf=10)
        xm_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)
    if calibration:
        x_clf = CalibratedClassifierCV(x_clf,
                                       method=calib_method)
        xm_clf = CalibratedClassifierCV(xm_clf, method=calib_method)

    return x_clf, xm_clf


def get_train_test_lists(crossfit, n, x):
    if crossfit < 2:
        train_test_list = [[np.arange(n), np.arange(n)]]
    else:
        kf = KFold(n_splits=crossfit)
        train_test_list = list()
        for train_index, test_index in kf.split(x):
            train_test_list.append([train_index, test_index])
    return train_test_list


def estimate_px(t, m, x, crossfit, classifier_x, classifier_xm):
    n = len(t)

    p_x, p_xm = [np.zeros(n) for h in range(2)]
    # compute propensity scores
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)

    train_test_list = get_train_test_lists(crossfit, n, x)

    for train_index, test_index in train_test_list:
        x_clf = classifier_x.fit(x[train_index, :], t[train_index])
        xm_clf = classifier_xm.fit(np.hstack((x, m))[train_index, :], t[train_index])
        p_x[test_index] = x_clf.predict_proba(x[test_index, :])[:, 1]
        p_xm[test_index] = xm_clf.predict_proba(np.hstack((x, m))[test_index, :])[:, 1]

    return p_x, p_xm

def get_y_m_classifiers(regularization, forest, calibration, calib_method):
    if regularization:
        alphas = ALPHAS
        cs = ALPHAS
    else:
        alphas = [TINY]
        cs = [np.inf]

    if not forest:
        clf_y = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        clf_m = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)
    else:
        clf_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
        clf_m = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)
    if calibration:
        clf_m = CalibratedClassifierCV(clf_m, method=calib_method)

    return clf_y, clf_m


def estimate_f_mu(t, m, x, y, crossfit, classifier_y, classifier_m, interaction):


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

    train_test_list = get_train_test_lists(crossfit, n, x)

    mu_11x, mu_10x, mu_01x, mu_00x = [np.zeros(n) for _ in range(4)]
    f_00x, f_01x, f_10x, f_11x = [np.zeros(n) for _ in range(4)]

    x_t_mr = get_interactions(interaction, x, t, mr)
    t_x = get_interactions(interaction, t, x)

    x_t1_m1 = get_interactions(interaction, x, t1, m1)
    x_t1_m0 = get_interactions(interaction, x, t1, m0)
    x_t0_m1 = get_interactions(interaction, x, t0, m1)
    x_t0_m0 = get_interactions(interaction, x, t0, m0)

    t0_x = get_interactions(interaction, t0, x)
    t1_x = get_interactions(interaction, t1, x)

    for train_index, test_index in train_test_list:

        clf_y = classifier_y.fit(x_t_mr[train_index, :], y[train_index])
        clf_m = classifier_m.fit(t_x[train_index, :], m.ravel()[train_index])
        mu_11x[test_index] = clf_y.predict(x_t1_m1[test_index, :])
        mu_10x[test_index] = clf_y.predict(x_t1_m0[test_index, :])
        mu_01x[test_index] = clf_y.predict(x_t0_m1[test_index, :])
        mu_00x[test_index] = clf_y.predict(x_t0_m0[test_index, :])
        f_00x[test_index] = clf_m.predict_proba(t0_x[test_index, :])[:, 0]
        f_01x[test_index] = clf_m.predict_proba(t0_x[test_index, :])[:, 1]
        f_10x[test_index] = clf_m.predict_proba(t1_x[test_index, :])[:, 0]
        f_11x[test_index] = clf_m.predict_proba(t1_x[test_index, :])[:, 1]

    f = f_00x, f_01x, f_10x, f_11x
    mu = mu_11x, mu_10x, mu_01x, mu_00x
    return f, mu

def get_y_m_x_classifiers(regularization, forest, calibration, calib_method):


    if regularization:
        alphas, cs = ALPHAS, ALPHAS
    else:
        alphas, cs = [TINY], [np.inf]

    # mu_tm, f_mtx, and p_x model fitting
    if not forest:
        y_clf = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        m_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)
        x_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)
    else:
        y_clf = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
        m_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)
        x_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)
    if calibration:
        m_clf = CalibratedClassifierCV(m_clf, method=calib_method)
        x_clf = CalibratedClassifierCV(x_clf, method=calib_method)

    cross_y_clf = RidgeCV(alphas=alphas, cv=CV_FOLDS)

    return y_clf, cross_y_clf, m_clf, x_clf

def estimate_f_mu_cross_mu(t, m, x, y, crossfit, classifier_y, classifier_cross_y, classifier_m, classifier_x, interaction):
    n = len(y)

    # Initialisation
    (
        p_x,  # P(T=1|X)
        f_00x,  # f(M=0|T=0,X)
        f_01x,  # f(M=0|T=1,X)
        f_10x,  # f(M=1|T=0,X)
        f_11x,  # f(M=1|T=1,X)
        f_m0x,  # f(M|T=0,X)
        f_m1x,  # f(M|T=1,X)
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
    ) = [np.zeros(n) for _ in range(17)]

    t0, m0 = np.zeros((n, 1)), np.zeros((n, 1))
    t1, m1 = np.ones((n, 1)), np.ones((n, 1))

    train_test_list = get_train_test_lists(crossfit, n, x)

    x_t_m = get_interactions(interaction, x, t, m)
    t_x = get_interactions(interaction, t, x)
    t0_x = get_interactions(interaction, t0, x)
    t1_x = get_interactions(interaction, t1, x)

    x_t1_m = get_interactions(interaction, x, t1, m)
    x_t0_m = get_interactions(interaction, x, t0, m)

    x_t0_m0 = get_interactions(interaction, x, t0, m0)
    x_t0_m1 = get_interactions(interaction, x, t0, m1)
    x_t1_m0 = get_interactions(interaction, x, t1, m0)
    x_t1_m1 = get_interactions(interaction, x, t1, m1)

    # Cross-fitting loop
    for train_index, test_index in train_test_list:
        # Index declaration
        test_ind = np.arange(len(test_index))
        ind_t0 = t[test_index] == 0

        # mu_tm, f_mtx, and p_x model fitting
        clf_y = classifier_y.fit(x_t_m[train_index, :], y[train_index])
        clf_m = classifier_m.fit(t_x[train_index, :], m[train_index])
        clf_x = classifier_x.fit(x[train_index, :], t[train_index])
        

        # predict P(T=1|X)
        p_x[test_index] = clf_x.predict_proba(x[test_index, :])[:, 1]

        # predict f(M=m|T=t,X)
        fm_0 = clf_m.predict_proba(t0_x[test_index, :])
        f_00x[test_index] = fm_0[:, 0]
        f_01x[test_index] = fm_0[:, 1]
        fm_1 = clf_m.predict_proba(t1_x[test_index, :])
        f_10x[test_index] = fm_1[:, 0]
        f_11x[test_index] = fm_1[:, 1]

        # predict f(M|T=t,X)
        f_m0x[test_index] = fm_0[test_ind, m[test_index]]
        f_m1x[test_index] = fm_1[test_ind, m[test_index]]

        # predict E[Y|T=t,M,X]
        mu_1mx[test_index] = clf_y.predict(x_t1_m[test_index, :])
        mu_0mx[test_index] = clf_y.predict(x_t0_m[test_index, :])

        # predict E[Y|T=t,M=m,X]
        mu_00x[test_index] = clf_y.predict(x_t0_m0[test_index, :])
        mu_01x[test_index] = clf_y.predict(x_t0_m1[test_index, :])
        mu_11x[test_index] = clf_y.predict(x_t1_m1[test_index, :])
        mu_10x[test_index] = clf_y.predict(x_t1_m0[test_index, :])

        # E[E[Y|T=1,M=m,X]|T=t,X] model fitting
        reg_y_t1m1_t0 = clone(classifier_cross_y).fit(
            x[test_index, :][ind_t0, :], mu_11x[test_index][ind_t0]
        )
        reg_y_t1m0_t0 = clone(classifier_cross_y).fit(
            x[test_index, :][ind_t0, :], mu_10x[test_index][ind_t0]
        )
        reg_y_t1m1_t1 = clone(classifier_cross_y).fit(
            x[test_index, :][~ind_t0, :], mu_11x[test_index][~ind_t0]
        )
        reg_y_t1m0_t1 = clone(classifier_cross_y).fit(
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
        reg_y_t0m1_t0 = clone(classifier_cross_y).fit(
            x[test_index, :][ind_t0, :], mu_01x[test_index][ind_t0]
        )
        reg_y_t0m0_t0 = clone(classifier_cross_y).fit(
            x[test_index, :][ind_t0, :], mu_00x[test_index][ind_t0]
        )
        reg_y_t0m1_t1 = clone(classifier_cross_y).fit(
            x[test_index, :][~ind_t0, :], mu_01x[test_index][~ind_t0]
        )
        reg_y_t0m0_t1 = clone(classifier_cross_y).fit(
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

    # f = f_00x, f_01x, f_10x, f_11x, f_m0x, f_m1x
    f = f_m0x, f_m1x
    # mu = mu_11x, mu_10x, mu_01x, mu_00x, mu_t0, mu_t1
    mu = mu_0mx, mu_1mx
    cross_mu = E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1

    return p_x, f, mu, cross_mu

def get_x_y_classifiers(regularization, forest, calib_method, random_state):


    # define regularization parameters
    if regularization:
        alphas = ALPHAS
        cs = ALPHAS
    else:
        alphas = [TINY]
        cs = [np.inf]

    if forest:
        clf_x = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)
        clf_xm = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)
        clf_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
        clf_cross_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
    else:
        clf_x = LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            random_state=random_state,
            Cs=cs,
            cv=CV_FOLDS,
        )
        clf_xm = LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            random_state=random_state,
            Cs=cs,
            cv=CV_FOLDS,
        )
        clf_y = LassoCV(alphas=alphas, cv=CV_FOLDS)
        clf_cross_y = LassoCV(alphas=alphas, cv=CV_FOLDS)

    if calib_method in {"sigmoid", "isotonic"}:
            clf_x = CalibratedClassifierCV(clf_x, method=calib_method)
            clf_xm = CalibratedClassifierCV(clf_xm, method=calib_method)

    return clf_x, clf_xm, clf_y, clf_cross_y

def estimate_px_mu_cross_mu(t, m, x, y, crossfit, classifier_x, classifier_xm, classifier_y, classifier_cross_y):

    n = len(y)

    # initialisation
    (
        p_x,  # P(T=1|X)
        p_xm,  # P(T=1|M,X)
        mu_1mx,  # E[Y|T=1,M,X]
        mu_1mx_nested,  # E[Y|T=1,M,X] predicted on train_nested set
        mu_0mx,  # E[Y|T=0,M,X]
        mu_0mx_nested,  # E[Y|T=0,M,X] predicted on train_nested set
        E_mu_t1_t0,  # E[E[Y|T=1,M,X]|T=0,X]
        E_mu_t0_t1,  # E[E[Y|T=0,M,X]|T=1,X]
        mu_1x,  # E[Y|T=1,X]
        mu_0x,  # E[Y|T=0,X]
    ) = [np.zeros(n) for _ in range(10)]

    xm = np.hstack((x, m))

    train_test_list = get_train_test_lists(crossfit, n, x)

    for train, test in train_test_list:
        # define test set
        train1 = train[t[train] == 1]
        train0 = train[t[train] == 0]

        train_mean, train_nested = np.array_split(train, 2)
        train_mean1 = train_mean[t[train_mean] == 1]
        train_mean0 = train_mean[t[train_mean] == 0]
        train_nested1 = train_nested[t[train_nested] == 1]
        train_nested0 = train_nested[t[train_nested] == 0]

        # predict P(T=1|X)
        classifier_x.fit(x[train], t[train])
        p_x[test] = classifier_x.predict_proba(x[test])[:, 1]

        # predict P(T=1|M,X)
        classifier_xm.fit(xm[train], t[train])
        p_xm[test] = classifier_xm.predict_proba(xm[test])[:, 1]

        # predict E[Y|T=1,M,X]
        clf_y1m = clone(classifier_y)
        clf_y1m.fit(xm[train_mean1], y[train_mean1])
        mu_1mx[test] = clf_y1m.predict(xm[test])
        mu_1mx_nested[train_nested] = clf_y1m.predict(xm[train_nested])

        # predict E[Y|T=0,M,X]
        clf_y0m = clone(classifier_y)
        clf_y0m.fit(xm[train_mean0], y[train_mean0])
        mu_0mx[test] = clf_y0m.predict(xm[test])
        mu_0mx_nested[train_nested] = clf_y0m.predict(xm[train_nested])

        # predict E[E[Y|T=1,M,X]|T=0,X]
        clf_cross_y1 = clone(classifier_cross_y)
        clf_cross_y1.fit(x[train_nested0], mu_1mx_nested[train_nested0])
        E_mu_t1_t0[test] = clf_cross_y1.predict(x[test])

        # predict E[E[Y|T=0,M,X]|T=1,X]
        clf_cross_y0 = clone(classifier_cross_y)
        clf_cross_y0.fit(x[train_nested1], mu_0mx_nested[train_nested1])
        E_mu_t0_t1[test] = clf_cross_y0.predict(x[test])

        # predict E[Y|T=1,X]
        clf_y1 = clone(classifier_y)
        clf_y1.fit(x[train1], y[train1])
        mu_1x[test] = clf_y1.predict(x[test])

        # predict E[Y|T=0,X]
        clf_y0 = clone(classifier_y)
        clf_y0.fit(x[train0], y[train0])
        mu_0x[test] = clf_y0.predict(x[test])

    p = p_x, p_xm
    mu = mu_1mx, mu_1mx_nested, mu_0mx, mu_0mx_nested, mu_1x, mu_0x
    cross_mu = E_mu_t0_t1, E_mu_t1_t0

    return p, mu, cross_mu