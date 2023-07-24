"""
the objective of this module is to implement estimators for mediation in
causal inference, purely in python, without R dependency
"""



from sklearn.linear_model import LogisticRegressionCV, LinearRegression, RidgeCV
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


ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5


def get_interactions(interaction, *args):
    """
    this function provides interaction terms between different groups of 
    variables (confounders, treatment, mediators)
    Inputs
    --------
    interaction     boolean
                    whether to compute interaction terms

    *args           flexible, one or several arrays
                    blocks of variables between which interactions should be
                    computed
    Returns
    --------
    Examples
    --------
    >>> x = np.arange(6).reshape(3, 2)
    >>> t = np.ones((3, 1))
    >>> m = 2 * np.ones((3, 1))
    >>> get_interactions(False, x, t, m)
    array([[0., 1., 1., 2.],
           [2., 3., 1., 2.],
           [4., 5., 1., 2.]])
    >>> get_interactions(True, x, t, m)
    array([[ 0.,  1.,  1.,  2.,  0.,  1.,  0.,  2.,  2.],
           [ 2.,  3.,  1.,  2.,  2.,  3.,  4.,  6.,  2.],
           [ 4.,  5.,  1.,  2.,  4.,  5.,  8., 10.,  2.]])
    """
    variables = args
    pre_inter_variables = np.hstack(variables)
    if not interaction:
        return pre_inter_variables
    else:
        new_cols = list()
        for i, var in enumerate(variables[:]):
            for j, var2 in enumerate(variables[i+1:]):
                for ii in range(var.shape[1]):
                    for jj in range(var2.shape[1]):
                        new_cols.append((var[:, ii] * var2[:, jj]).reshape(-1, 1))
        new_vars = np.hstack(new_cols)
        result = np.hstack(variables + (new_vars,))
        return result



def huber_IPW(y, t, m, x, w, z, trim, logit, regularization=True, forest=False,
              crossfit=0, clip=0.01, calibration=True, calib_method='sigmoid'):
    """
    IPW estimator presented in
    HUBER, Martin. Identifying causal mechanisms (primarily) based on inverse
    probability weighting. Journal of Applied Econometrics, 2014,
    vol. 29, no 6, p. 920-943.

    results has 6 values
    - total effect
    - direct effect treated (\theta(1))
    - direct effect non treated (\theta(0))
    - indirect effect treated (\delta(1))
    - indirect effect untreated (\delta(0))
    - number of used observations (non trimmed)

    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples, n_features_mediator)
            mediator value for each unit, can be continuous or binary, and
            multi-dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    w       array-like, shape (n_samples, n_features_mediator_bis)
            other mediator value (w causes m)
            can be continuous or binary, and multi-dimensional

    z       Optional instrumental variable(s)
            not implemented yet, mentioned to mimick the signature of
            the medweight function in the R package causalweight

    trim    float
            Trimming rule for discarding observations with extreme propensity
            scores. In the absence of post-treatment confounders (w=NULL),
            observations with Pr(D=1|M,X)<trim or Pr(D=1|M,X)>(1-trim) are
            dropped. In the presence of post-treatment confounders
            (w is defined), observations with Pr(D=1|M,W,X)<trim or
            Pr(D=1|M,W,X)>(1-trim) are dropped.

    logit   boolean
            whether logit or pobit regression is used for propensity score
            legacy from the R package, here only logit is implemented

    regularization boolean, default True
                   whether to use regularized models (logistic or 
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5

    forest  boolean, default False
            whether to use a random forest model to estimate the propensity
            scores instead of logistic regression

    crossfit integer, default 0
             number of folds for cross-fitting

    clip    float
            limit to clip for numerical stability (min=clip, max=1-clip)
    """
    if regularization:
        cs = ALPHAS
    else:
        cs = [np.inf]
    n = len(t)
    if crossfit < 2:
        train_test_list = [[np.arange(n), np.arange(n)]]
    else:
        kf = KFold(n_splits=crossfit)
        train_test_list = list()
        for train_index, test_index in kf.split(x):
            train_test_list.append([train_index, test_index])
    if w is None:
        if z is not None:
            raise NotImplementedError
        else:
            p_x, p_xm = [np.zeros(n) for h in range(2)]
            # compute propensity scores
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            if len(m.shape) == 1:
                m = m.reshape(-1, 1)
            for train_index, test_index in train_test_list:
                if not forest:
                    rf_x_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
                        .fit(x[train_index, :], t[train_index])
                    rf_xm_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
                        .fit(np.hstack((x, m))[train_index, :], t[train_index])
                else:
                    rf_x_clf = RandomForestClassifier(n_estimators=100,
                                                      min_samples_leaf=10)\
                        .fit(x[train_index, :], t[train_index])
                    rf_xm_clf = RandomForestClassifier(n_estimators=100,
                                                       min_samples_leaf=10)\
                        .fit(np.hstack((x, m))[train_index, :], t[train_index])
                if calibration:
                    p_x_clf = CalibratedClassifierCV(rf_x_clf,
                                                     method=calib_method)\
                        .fit(x[train_index, :], t[train_index])
                    p_xm_clf = CalibratedClassifierCV(rf_xm_clf,
                                                      method=calib_method)\
                        .fit(np.hstack((x, m))[train_index, :], t[train_index])
                else:
                    p_x_clf = rf_x_clf
                    p_xm_clf = rf_xm_clf
                p_x[test_index] = p_x_clf.predict_proba(x[test_index, :])[:, 1]
                p_xm[test_index] = p_xm_clf.predict_proba(
                    np.hstack((x, m))[test_index, :])[:, 1]

            # trimming. Following causal weight code, not sure I understand
            # why we trim only on p_xm and not on p_x
            ind = ((p_xm > trim) & (p_xm < (1 - trim)))
            y, t, p_x, p_xm = y[ind], t[ind], p_x[ind], p_xm[ind]

            # note on the names, ytmt' = Y(t, M(t')), the treatment needs to be
            # binary but not the mediator
            p_x = np.clip(p_x, clip, 1 - clip)
            p_xm = np.clip(p_xm, clip, 1 - clip)

            y1m1 = np.sum(y * t / p_x) / np.sum(t / p_x)
            y1m0 = np.sum(y * t * (1 - p_xm) / (p_xm * (1 - p_x))) /\
                np.sum(t * (1 - p_xm) / (p_xm * (1 - p_x)))
            y0m0 = np.sum(y * (1 - t) / (1 - p_x)) /\
                np.sum((1 - t) / (1 - p_x))
            y0m1 = np.sum(y * (1 - t) * p_xm / ((1 - p_xm) * p_x)) /\
                np.sum((1 - t) * p_xm / ((1 - p_xm) * p_x))

            return(y1m1 - y0m0,
                   y1m1 - y0m1,
                   y1m0 - y0m0,
                   y1m1 - y1m0,
                   y0m1 - y0m0,
                   np.sum(ind))

    else:
        if z is not None:
            raise NotImplementedError
        else:
            p_x, p_wx, p_xmw = [np.zeros(n) for h in range(3)]
            # compute propensity scores
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)
            if len(m.shape) == 1:
                m = m.reshape(-1, 1)
            if len(w.shape) == 1:
                w = w.reshape(-1, 1)
            for train_index, test_index in train_test_list:
                if not forest:
                    rf_x_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
                        .fit(x[train_index, :], t[train_index])
                    rf_xw_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
                        .fit(np.hstack((x, w))[train_index, :], t[train_index])
                    rf_xmw_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
                        .fit(np.hstack((x, m, w))[train_index, :],
                             t[train_index])
                else:
                    rf_x_clf = RandomForestClassifier(n_estimators=100,
                                                      min_samples_leaf=10)
                    rf_xw_clf = RandomForestClassifier(n_estimators=100,
                                                       min_samples_leaf=10)
                    rf_xmw_clf = RandomForestClassifier(n_estimators=100,
                                                        min_samples_leaf=10)
                if calibration:
                    p_x_clf = CalibratedClassifierCV(rf_x_clf,
                                                     method=calib_method)\
                        .fit(x[train_index, :], t[train_index])
                    p_wx_clf = CalibratedClassifierCV(rf_xw_clf,
                                                      method=calib_method)\
                        .fit(np.hstack((x, w))[train_index, :], t[train_index])
                    p_xmw_clf = CalibratedClassifierCV(rf_xmw_clf,
                                                       method=calib_method)\
                        .fit(np.hstack((x, m, w))[train_index, :],
                             t[train_index])
                else:
                    p_x_clf = rf_x_clf
                    p_wx_clf = rf_xw_clf
                    p_xmw_clf = rf_xmw_clf
                p_x[test_index] = p_x_clf.predict_proba(x[test_index, :])[:, 1]
                p_wx[test_index] = p_wx_clf.predict_proba(
                    np.hstack((x, w))[test_index, :])[:, 1]
                p_xmw[test_index] = p_xmw_clf.predict_proba(
                    np.hstack((x, m, w))[test_index, :])[:, 1]

            # trimming. Following causal weight code, not sure I understand
            # why we trim only on p_xm and not on p_x
            ind = ((p_xmw > trim) & (p_xmw < (1 - trim)))
            y, t, p_x, p_wx, p_xmw = (y[ind], t[ind], p_x[ind], p_wx[ind],
                                      p_xmw[ind])

            p_x = np.clip(p_x, clip, 1 - clip)
            p_xmw = np.clip(p_xmw, clip, 1 - clip)
            p_wx = np.clip(p_wx, clip, 1 - clip)

            # computation of effects
            y1m1 = np.sum(y * t / p_x) / np.sum(t / p_x)
            y1m0 = np.sum(y * t * (1 - p_xmw) / ((1 - p_x) * p_xmw)) /\
                np.sum(t * (1 - p_xmw) / ((1 - p_x) * p_xmw))
            y0m0 = np.sum(y * (1 - t) / (1 - p_x)) /\
                np.sum((1 - t) / (1 - p_x))
            y0m1 = np.sum(y * (1 - t) * p_xmw / (p_x * (1 - p_xmw))) /\
                np.sum((1 - t) * p_xmw / (p_x * (1 - p_xmw)))
            y1m0p = np.sum(y * t / p_xmw * (1 - p_xmw) / (1 - p_wx) * p_wx / p_x)/\
                np.sum(t / p_xmw * (1 - p_xmw) / (1 - p_wx) * p_wx / p_x)
            y0m1p = np.sum(y * (1 - t) / (1 - p_xmw) * p_xmw / p_wx * (1 - p_wx) / (1 - p_x)) /\
                np.sum((1 - t) / (1 - p_xmw) * p_xmw / p_wx * (1 - p_wx) / (1 - p_x))

            return(y1m1 - y0m0,
                   y1m1 - y0m1,
                   y1m0 - y0m0,
                   y1m1 - y1m0p,
                   y0m1p - y0m0,
                   len(y) - np.sum(ind))


def ols_mediation(y, t, m, x, interaction=False, regularization=True):
    """
    found an R implementation https://cran.r-project.org/package=regmedint

    implements very simple model of mediation
    Y ~ X + T + M
    M ~ X + T
    estimation method is product of coefficients

    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples)
            mediator value for each unit, can be continuous or binary, and
            is necessary in 1D

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    interaction boolean, default=False
                whether to include interaction terms in the model
                not implemented here, just for compatibility of signature
                function

    regularization boolean, default True
                   whether to use regularized models (logistic or
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5

    """
    if regularization:
        alphas = ALPHAS
    else:
        alphas = [0.0]
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)
    if len(t.shape) == 1:
        t = t.reshape(-1, 1)
    coef_t_m = np.zeros(m.shape[1])
    for i in range(m.shape[1]):
        m_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)\
            .fit(np.hstack((x, t)), m[:, i])
        coef_t_m[i] = m_reg.coef_[-1]
    y_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)\
        .fit(np.hstack((x, t, m)), y.ravel())

    # return total, direct and indirect effect
    direct_effect = y_reg.coef_[x.shape[1]]
    indirect_effect = sum(y_reg.coef_[x.shape[1] + 1:] * coef_t_m)
    return [direct_effect + indirect_effect,
            direct_effect,
            direct_effect,
            indirect_effect,
            indirect_effect,
            None]


def g_computation(y, t, m, x, interaction=False, forest=False,
                  crossfit=0, calibration=True, regularization=True,
                  calib_method='sigmoid'):
    """
    m is binary !!!

    implementation of the g formula for mediation

    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary and uni-
            dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    interaction boolean, default=False
                whether to include interaction terms in the model
                interactions are terms XT, TM, MX

    forest  boolean, default False
            whether to use a random forest model to estimate the propensity
            scores instead of logistic regression, and outcome model instead
            of linear regression

    crossfit integer, default 0
             number of folds for cross-fitting

    regularization boolean, default True
                   whether to use regularized models (logistic or
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5
    """
    if regularization:
        alphas = ALPHAS
        cs = ALPHAS
    else:
        alphas = [0.0]
        cs = [np.inf]
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

    if crossfit < 2:
        train_test_list = [[np.arange(n), np.arange(n)]]
    else:
        kf = KFold(n_splits=crossfit)
        train_test_list = list()
        for train_index, test_index in kf.split(x):
            train_test_list.append([train_index, test_index])
    mu_11x, mu_10x, mu_01x, mu_00x, f_00x, f_01x, f_10x, f_11x = \
        [np.zeros(n) for h in range(8)]

    for train_index, test_index in train_test_list:
        if not forest:
            y_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)\
                .fit(get_interactions(interaction, x, t, mr)[train_index, :], y[train_index])
            pre_m_prob = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
                .fit(get_interactions(interaction, t, x)[train_index, :], m.ravel()[train_index])
        else:
            y_reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)\
                .fit(get_interactions(interaction, x, t, mr)[train_index, :], y[train_index])
            pre_m_prob = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)\
                .fit(get_interactions(interaction, t, x)[train_index, :], m.ravel()[train_index])
        if calibration:
            m_prob = CalibratedClassifierCV(pre_m_prob, method=calib_method)\
                .fit(get_interactions(
                    interaction, t, x)[train_index, :], m.ravel()[train_index])
        else:
            m_prob = pre_m_prob
        mu_11x[test_index] = y_reg.predict(get_interactions(interaction, x, t1, m1)[test_index, :])
        mu_10x[test_index] = y_reg.predict(get_interactions(interaction, x, t1, m0)[test_index, :])
        mu_01x[test_index] = y_reg.predict(get_interactions(interaction, x, t0, m1)[test_index, :])
        mu_00x[test_index] = y_reg.predict(get_interactions(interaction, x, t0, m0)[test_index, :])
        f_00x[test_index] = m_prob.predict_proba(get_interactions(interaction, t0, x)[test_index, :])[:, 0]
        f_01x[test_index] = m_prob.predict_proba(get_interactions(interaction, t0, x)[test_index, :])[:, 1]
        f_10x[test_index] = m_prob.predict_proba(get_interactions(interaction, t1, x)[test_index, :])[:, 0]
        f_11x[test_index] = m_prob.predict_proba(get_interactions(interaction, t1, x)[test_index, :])[:, 1]

    direct_effect_i1 = mu_11x - mu_01x
    direct_effect_i0 = mu_10x - mu_00x
    direct_effect_treated = (direct_effect_i1 * f_11x + direct_effect_i0 * f_10x).sum() / n
    direct_effect_control = (direct_effect_i1 * f_01x + direct_effect_i0 * f_00x).sum() / n
    indirect_effect_i1 = f_11x - f_01x
    indirect_effect_i0 = f_10x - f_00x
    indirect_effect_treated = (indirect_effect_i1 * mu_11x + indirect_effect_i0 * mu_10x).sum() / n
    indirect_effect_control = (indirect_effect_i1 * mu_01x + indirect_effect_i0 * mu_00x).sum() / n
    total_effect = direct_effect_control + indirect_effect_treated

    return [total_effect,
            direct_effect_treated,
            direct_effect_control,
            indirect_effect_treated,
            indirect_effect_control,
            None]


def alternative_estimator(y, t, m, x, regularization=True):
    """
    presented in
    HUBER, Martin, LECHNER, Michael, et MELLACE, Giovanni.
    The finite sample performance of estimators for mediation analysis under
    sequential conditional independence.
    Journal of Business & Economic Statistics, 2016, vol. 34, no 1, p. 139-160.
    section 3.2.2

    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary and uni-
            dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    regularization boolean, default True
                   whether to use regularized models (logistic or
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5
    """
    if regularization:
        alphas = ALPHAS
    else:
        alphas = [0.0]
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)
    treated = (t == 1)

    # computation of direct effect
    y_treated_reg_m = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(np.hstack((x[treated], m[treated])), y[treated])
    y_ctrl_reg_m = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(np.hstack((x[~treated], m[~treated])), y[~treated])
    direct_effect = np.sum(y_treated_reg_m.predict(np.hstack((x, m))) - y_ctrl_reg_m.predict(np.hstack((x, m)))) / len(y)

    # computation of total effect
    y_treated_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(x[treated], y[treated])
    y_ctrl_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(x[~treated], y[~treated])
    total_effect = np.sum(y_treated_reg.predict(x) - y_ctrl_reg.predict(x)) / len(y)

    # computation of indirect effect
    indirect_effect = total_effect - direct_effect

    return [total_effect,
            direct_effect,
            direct_effect,
            indirect_effect,
            indirect_effect,
            None]


def multiply_robust_efficient(y, t, m, x, interaction=False,
                              forest=False, crossfit=0, clip=0.01,
                              regularization=True, calibration=True,
                              calib_method='sigmoid'):
    """
    presented in Eric J. Tchetgen Tchetgen. Ilya Shpitser.
    "Semiparametric theory for causal mediation analysis: Efficiency bounds,
    multiple robustness and sensitivity analysis."
    Ann. Statist. 40 (3) 1816 - 1845, June 2012.
    https://doi.org/10.1214/12-AOS990

    m is binary !!!

    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary and uni-
            dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    interaction boolean, default=False
                whether to include interaction terms in the model
                interactions are terms XT, TM, MX

    penalty string {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
            passed to the scikit learn sklearn.linear_model.LogisticRegression
            model. Was used to mimick the R implementation without penalty in
            some cases, to evaluate regularization bias

    forest  boolean, default False
            whether to use a random forest model to estimate the propensity
            scores instead of logistic regression, and outcome model instead
            of linear regression

    crossfit integer, default 0
             number of folds for cross-fitting

    clip    float
            limit to clip for numerical stability (min=clip, max=1-clip)

    regularization boolean, default True
                   whether to use regularized models (logistic or
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5

    """
    if regularization:
        alphas = ALPHAS
        cs = ALPHAS
    else:
        alphas = [0.0]
        cs = [np.inf]
    n = len(y)
    ind = np.arange(n)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        mr = m.reshape(-1, 1)
    else:
        mr = np.copy(m)
        m = m.ravel()
    if len(t.shape) == 1:
        tr = t.reshape(-1, 1)
    t0 = np.zeros((n, 1))
    t1 = np.ones((n, 1))
    m0 = np.zeros((n, 1))
    m1 = np.ones((n, 1))

    if crossfit < 2:
        train_test_list = [[np.arange(n), np.arange(n)]]
    else:
        kf = KFold(n_splits=crossfit)
        train_test_list = list()
        for train_index, test_index in kf.split(x):
            train_test_list.append([train_index, test_index])
    p_x, diff_mu_m0, diff_mu_m1, f_00x, f_01x, theta_0x, f_m0x, f_m1x, mu_i, mu_t1, mu_t0, mu_t1m1, mu_t1m0, f_10x, f_11x, psi_0x, psi_1x = \
        [np.zeros(n) for h in range(17)]

    for train_index, test_index in train_test_list:
        test_ind = np.arange(len(test_index))
        if not forest:
            y_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)\
                .fit(get_interactions(interaction, x, tr, mr)[train_index, :],
                     y[train_index])
            pre_m_prob = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
                .fit(get_interactions(interaction, tr, x)[train_index, :],
                     m[train_index])
            pre_p_x_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
                .fit(x[train_index, :], t[train_index])
        else:
            y_reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)\
                .fit(get_interactions(interaction, x, tr, mr)[train_index, :], y[train_index])
            pre_m_prob = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)\
                .fit(get_interactions(interaction, tr, x)[train_index, :], m[train_index])
            pre_p_x_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)\
                .fit(x[train_index, :], t[train_index])
        if calibration:
            m_prob = CalibratedClassifierCV(pre_m_prob, method=calib_method)\
                .fit(get_interactions(interaction, tr, x)[train_index, :], m[train_index])
            p_x_clf = CalibratedClassifierCV(pre_p_x_clf, method=calib_method)\
                .fit(x[train_index, :], t[train_index])
        else:
            m_prob = pre_m_prob
            p_x_clf = pre_p_x_clf

        p_x[test_index] = p_x_clf.predict_proba(x[test_index, :])[:, 1]

        ind_t0 = tr[test_index, 0] == 0
        diff_mu_m0[test_index] = y_reg.predict(get_interactions(interaction, x, t1, m0)[test_index, :]) -\
            y_reg.predict(get_interactions(interaction, x, t0, m0)[test_index, :])
        diff_mu_m1[test_index] = y_reg.predict(get_interactions(interaction, x, t1, m1)[test_index, :]) -\
            y_reg.predict(get_interactions(interaction, x, t0, m1)[test_index, :])
        diff_mu_m0_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(x[test_index, :][ind_t0, :], diff_mu_m0[test_index][ind_t0])
        diff_mu_m1_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(x[test_index, :][ind_t0, :], diff_mu_m1[test_index][ind_t0])
        f_00x[test_index] = m_prob.predict_proba(get_interactions(interaction, t0, x)[test_index, :])[:, 0]
        f_01x[test_index] = m_prob.predict_proba(get_interactions(interaction, t0, x)[test_index, :])[:, 1]
        theta_0x[test_index] = diff_mu_m0_reg.predict(x[test_index, :]) * f_00x[test_index] +\
            diff_mu_m1_reg.predict(x[test_index, :]) * f_01x[test_index]

        f_m0x[test_index] = m_prob.predict_proba(get_interactions(interaction, t0, x)[test_index, :])[test_ind, m[test_index]]
        f_m1x[test_index] = m_prob.predict_proba(get_interactions(interaction, t1, x)[test_index, :])[test_ind, m[test_index]]

        mu_i[test_index] = y_reg.predict(get_interactions(interaction, x, tr, mr)[test_index, :])
        mu_t1[test_index] = y_reg.predict(get_interactions(interaction, x, t1, mr)[test_index, :])
        mu_t0[test_index] = y_reg.predict(get_interactions(interaction, x, t0, mr)[test_index, :])

        mu_t1m1[test_index] = y_reg.predict(get_interactions(interaction, x, t1, m1)[test_index, :])
        mu_t1m0[test_index] = y_reg.predict(get_interactions(interaction, x, t1, m0)[test_index, :])
        reg_y_t1m1_t0 = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(x[test_index, :][ind_t0, :], mu_t1m1[test_index][ind_t0])
        reg_y_t1m0_t0 = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(x[test_index, :][ind_t0, :], mu_t1m0[test_index][ind_t0])
        reg_y_t1m1_t1 = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(x[test_index, :][~ind_t0, :], mu_t1m1[test_index][~ind_t0])
        reg_y_t1m0_t1 = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(x[test_index, :][~ind_t0, :], mu_t1m0[test_index][~ind_t0])
        f_10x[test_index] = m_prob.predict_proba(get_interactions(interaction, t1, x)[test_index, :])[:, 0]
        f_11x[test_index] = m_prob.predict_proba(get_interactions(interaction, t1, x)[test_index, :])[:, 1]

        psi_0x[test_index] = reg_y_t1m0_t0.predict(x[test_index, :]) *\
            f_00x[test_index] + reg_y_t1m1_t0.predict(x[test_index, :]) *\
            f_01x[test_index]
        psi_1x[test_index] = reg_y_t1m0_t1.predict(x[test_index, :]) *\
            f_10x[test_index] + reg_y_t1m1_t1.predict(x[test_index, :]) *\
            f_11x[test_index]

    p_x = np.clip(p_x, clip, 1 - clip)
    f_m0x = np.clip(f_m0x, clip, 1 - clip)
    f_m1x = np.clip(f_m1x, clip, 1 - clip)

    direct_effect_control = ((t * f_m0x / (p_x * f_m1x) - (1 - t) / (1 - p_x))
                             * (y - mu_i)
                             + (1 - t) / (1 - p_x) * (mu_t1 - mu_t0 - theta_0x)
                             + theta_0x).sum() / n

    indirect_effect_treated = (t / p_x * (y - psi_1x -
                                          f_m0x / f_m1x * (y - mu_t1)) -
                               (1 - t) / (1 - p_x) * (mu_t1 - psi_0x) +
                               psi_1x - psi_0x).sum() / n
    return [direct_effect_control + indirect_effect_treated,
            np.nan,
            direct_effect_control,
            indirect_effect_treated,
            np.nan,
            None]
