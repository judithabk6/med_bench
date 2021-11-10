"""
the objective of this script is to implement estimators for mediation in
causal inference, simulate data, and evaluate and compare estimators
"""

# first step, run r code to have the original implementation by Huber
# using rpy2 to have the same data in R and python...

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri, numpy2ri
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from numpy.random import default_rng
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import bernoulli
from scipy.special import expit

from itertools import combinations
from sklearn.model_selection import KFold

pandas2ri.activate()
numpy2ri.activate()

causalweight = rpackages.importr('causalweight')
mediation = rpackages.importr('mediation')
Rstats = rpackages.importr('stats')
base = rpackages.importr('base')
brc = rpackages.importr('bartCause')
grf = rpackages.importr('grf')
plmed = rpackages.importr('plmed')


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


def plain_IPW(y, t, x, trim=0.01):
    """
    plain IPW estimator without mediation
    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    p_x_clf = LogisticRegression().fit(x, t)
    p_x = p_x_clf.predict_proba(x)[:, 1]
    # trimming
    p_x[p_x < trim] = trim
    p_x[p_x > 1 - trim] = 1 - trim
    y1m1 = np.sum(y * t / p_x) / np.sum(t / p_x)
    y0m0 = np.sum(y * (1 - t) / (1 - p_x)) /\
        np.sum((1 - t) / (1 - p_x))
    return y1m1 - y0m0


def AIPW(y, t, m, x, clip=0.01, forest=False, crossfit=0, forest_r=False,
         alpha_ridge=0.00001, penalty='l2'):
    """
    AIPW estimator
    """
    n = len(y)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if crossfit < 2:
        train_test_list = [[np.arange(n), np.arange(n)]]
    else:
        kf = KFold(n_splits=crossfit)
        train_test_list = list()
        for train_index, test_index in kf.split(x):
            train_test_list.append([train_index, test_index])

    if not forest_r:
        mu_1x, mu_0x, e_x = [np.zeros(n) for h in range(3)]

        for train_index, test_index in train_test_list:
            treated_train_index = np.array(
                list(set(train_index).intersection(np.where(t == 1)[0])))
            control_train_index = np.array(
                list(set(train_index).intersection(np.where(t == 0)[0])))
            if not forest:
                y_reg_treated = Ridge(alpha=alpha_ridge)\
                    .fit(x[treated_train_index, :], y[treated_train_index])
                y_reg_control = Ridge(alpha=alpha_ridge)\
                    .fit(x[control_train_index, :], y[control_train_index])
                t_prob = LogisticRegression(penalty=penalty)\
                    .fit(x[train_index, :], t[train_index])
            else:
                y_reg_treated = RandomForestRegressor(max_depth=3,
                                                      min_samples_leaf=5)\
                    .fit(x[treated_train_index, :], y[treated_train_index])
                y_reg_control = RandomForestRegressor(max_depth=3,
                                                      min_samples_leaf=5)\
                    .fit(x[control_train_index, :], y[control_train_index])
                t_prob = CalibratedClassifierCV(
                    RandomForestClassifier(max_depth=3, min_samples_leaf=5))\
                    .fit(x[train_index, :], t[train_index])
            mu_1x[test_index] = y_reg_treated.predict(x[test_index, :])
            mu_0x[test_index] = y_reg_control.predict(x[test_index, :])
            e_x[test_index] = t_prob.predict_proba(x[test_index, :])[:, 1]
        e_x = np.clip(e_x, clip, 1 - clip)
        total_effect = np.mean(mu_1x - mu_0x + t * (y - mu_1x) / e_x -
                               (1 - t) * (y - mu_0x) / (1 - e_x))
    else:
        x_r, t_r, y_r = [_convert_array_to_R(uu) for uu in (x, t, y)]
        cf = grf.causal_forest(x_r, y_r, t_r, num_trees=500)
        total_effect = grf.average_treatment_effect(cf)[0]
    return [total_effect] + [None] * 5


def huber_IPW(y, t, m, x, w, z, trim, logit, penalty='l2', forest=False,
              crossfit=0, clip=0.01, calibration=True):
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

    penalty string {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
            passed to the scikit learn sklearn.linear_model.LogisticRegression
            model. Was used to mimick the R implementation without penalty in
            some cases, to evaluate regularization bias

    forest  boolean, default False
            whether to use a random forest model to estimate the propensity
            scores instead of logistic regression

    crossfit integer, default 0
             number of folds for cross-fitting

    clip    float
            limit to clip for numerical stability (min=clip, max=1-clip)
    """
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
                    rf_x_clf = LogisticRegression(penalty=penalty)\
                        .fit(x[train_index, :], t[train_index])
                    rf_xm_clf = LogisticRegression(penalty=penalty)\
                        .fit(np.hstack((x, m))[train_index, :], t[train_index])
                else:
                    rf_x_clf = RandomForestClassifier(n_estimators=100,
                                                      min_samples_leaf=10)\
                        .fit(x[train_index, :], t[train_index])
                    rf_xm_clf = RandomForestClassifier(n_estimators=100,
                                                       min_samples_leaf=10)\
                        .fit(np.hstack((x, m))[train_index, :], t[train_index])
                if calibration:
                    p_x_clf = CalibratedClassifierCV(rf_x_clf)\
                        .fit(x[train_index, :], t[train_index])
                    p_xm_clf = CalibratedClassifierCV(rf_xm_clf)\
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
                    rf_x_clf = LogisticRegression(penalty=penalty)\
                        .fit(x[train_index, :], t[train_index])
                    rf_xw_clf = LogisticRegression(penalty=penalty)\
                        .fit(np.hstack((x, w))[train_index, :], t[train_index])
                    rf_xmw_clf = LogisticRegression(penalty=penalty)\
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
                    p_x_clf = CalibratedClassifierCV(rf_x_clf)\
                        .fit(x[train_index, :], t[train_index])
                    p_wx_clf = CalibratedClassifierCV(rf_xw_clf)\
                        .fit(np.hstack((x, w))[train_index, :], t[train_index])
                    p_xmw_clf = CalibratedClassifierCV(rf_xmw_clf)\
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


def ols_mediation(y, t, m, x, interaction=False, alpha_ridge=0.00001):
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
                not implemented here, just for compatbility of signature function

    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)
    if len(t.shape) == 1:
        t = t.reshape(-1, 1)
    coef_t_m = np.zeros(m.shape[1])
    for i in range(m.shape[1]):
        m_reg = Ridge(alpha=alpha_ridge).fit(np.hstack((x, t)), m[:, i])
        coef_t_m[i] = m_reg.coef_[-1]
    y_reg = Ridge(alpha=alpha_ridge).fit(np.hstack((x, t, m)), y.ravel())

    # return total, direct and indirect effect
    direct_effect = y_reg.coef_[x.shape[1]]
    indirect_effect = sum(y_reg.coef_[x.shape[1] + 1:] * coef_t_m)
    return [direct_effect + indirect_effect,
            direct_effect,
            direct_effect,
            indirect_effect,
            indirect_effect,
            None]


def g_computation(y, t, m, x, interaction=False, penalty='l2', forest=False,
                  crossfit=0, alpha_ridge=0.00001, calibration=True):
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
            y_reg = Ridge(alpha=alpha_ridge).fit(get_interactions(interaction, x, t, mr)[train_index, :], y[train_index])
            pre_m_prob = LogisticRegression(penalty=penalty).fit(get_interactions(interaction, t, x)[train_index, :], m.ravel()[train_index])
        else:
            y_reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)\
                .fit(get_interactions(interaction, x, t, mr)[train_index, :], y[train_index])
            pre_m_prob = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)\
                .fit(get_interactions(interaction, t, x)[train_index, :], m.ravel()[train_index])
        if calibration:
            m_prob = CalibratedClassifierCV(pre_m_prob).fit(get_interactions(interaction, t, x)[train_index, :], m.ravel()[train_index])
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


def alternative_estimator(y, t, m, x, alpha_ridge=0.00001):
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

    """
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)
    treated = (t == 1)

    # computation of direct effect
    y_treated_reg_m = Ridge(alpha=alpha_ridge).fit(np.hstack((x[treated], m[treated])), y[treated])
    y_ctrl_reg_m = Ridge(alpha=alpha_ridge).fit(np.hstack((x[~treated], m[~treated])), y[~treated])
    direct_effect = np.sum(y_treated_reg_m.predict(np.hstack((x, m))) - y_ctrl_reg_m.predict(np.hstack((x, m)))) / len(y)

    # computation of total effect
    y_treated_reg = Ridge(alpha=alpha_ridge).fit(x[treated], y[treated])
    y_ctrl_reg = Ridge(alpha=alpha_ridge).fit(x[~treated], y[~treated])
    total_effect = np.sum(y_treated_reg.predict(x) - y_ctrl_reg.predict(x)) / len(y)

    # computation of indirect effect
    indirect_effect = total_effect - direct_effect

    return [total_effect,
            direct_effect,
            direct_effect,
            indirect_effect,
            indirect_effect,
            None]


def multiply_robust_efficient(y, t, m, x, interaction=False, penalty='l2',
                              forest=False, crossfit=0, clip=0.01,
                              alpha_ridge=0.00001, calibration=True):
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

    """
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
            y_reg = Ridge(alpha=alpha_ridge)\
                .fit(get_interactions(interaction, x, tr, mr)[train_index, :],
                     y[train_index])
            pre_m_prob = LogisticRegression(penalty=penalty)\
                .fit(get_interactions(interaction, tr, x)[train_index, :],
                     m[train_index])
            pre_p_x_clf = LogisticRegression(penalty=penalty)\
                .fit(x[train_index, :], t[train_index])
        else:
            y_reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)\
                .fit(get_interactions(interaction, x, tr, mr)[train_index, :], y[train_index])
            pre_m_prob = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)\
                .fit(get_interactions(interaction, tr, x)[train_index, :], m[train_index])
            pre_p_x_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=10)\
                .fit(x[train_index, :], t[train_index])
        if calibration:
            m_prob = CalibratedClassifierCV(pre_m_prob)\
                .fit(get_interactions(interaction, tr, x)[train_index, :], m[train_index])
            p_x_clf = CalibratedClassifierCV(pre_p_x_clf)\
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
        diff_mu_m0_reg = Ridge(alpha=alpha_ridge).fit(x[test_index, :][ind_t0, :], diff_mu_m0[test_index][ind_t0])
        diff_mu_m1_reg = Ridge(alpha=alpha_ridge).fit(x[test_index, :][ind_t0, :], diff_mu_m1[test_index][ind_t0])
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
        reg_y_t1m1_t0 = Ridge(alpha=alpha_ridge).fit(x[test_index, :][ind_t0, :], mu_t1m1[test_index][ind_t0])
        reg_y_t1m0_t0 = Ridge(alpha=alpha_ridge).fit(x[test_index, :][ind_t0, :], mu_t1m0[test_index][ind_t0])
        reg_y_t1m1_t1 = Ridge(alpha=alpha_ridge).fit(x[test_index, :][~ind_t0, :], mu_t1m1[test_index][~ind_t0])
        reg_y_t1m0_t1 = Ridge(alpha=alpha_ridge).fit(x[test_index, :][~ind_t0, :], mu_t1m0[test_index][~ind_t0])
        f_10x[test_index] = m_prob.predict_proba(get_interactions(interaction, t1, x)[test_index, :])[:, 0]
        f_11x[test_index] = m_prob.predict_proba(get_interactions(interaction, t1, x)[test_index, :])[:, 1]

        psi_0x[test_index] = reg_y_t1m0_t0.predict(x[test_index, :]) * f_00x[test_index] +\
            reg_y_t1m1_t0.predict(x[test_index, :]) * f_01x[test_index]
        psi_1x[test_index] = reg_y_t1m0_t1.predict(x[test_index, :]) * f_10x[test_index] +\
            reg_y_t1m1_t1.predict(x[test_index, :]) * f_11x[test_index]

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


def r_mediate(y, t, m, x, interaction=False):
    """
    This function calls the R function mediate from the package mediation
    (https://cran.r-project.org/package=mediation)
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
    """
    m = m.ravel()
    var_names = [[y, 'y'],
                 [t, 't'],
                 [m, 'm'],
                 [x, 'x']]
    df_list = list()
    for var, name in var_names:
        if len(var.shape) > 1:
            var_dim = var.shape[1]
            col_names = ['{}_{}'.format(name, i) for i in range(var_dim)]
            sub_df = pd.DataFrame(var, columns=col_names)
        else:
            sub_df = pd.DataFrame(var, columns=[name])
        df_list.append(sub_df)
        df = pd.concat(df_list, axis=1)
    m_features = [c for c in df.columns if ('y' not in c) and ('m' not in c)]
    y_features = [c for c in df.columns if ('y' not in c)]
    if not interaction:
        m_formula = 'm ~ ' + ' + '.join(m_features)
        y_formula = 'y ~ ' + ' + '.join(y_features)
    else:
        m_formula = 'm ~ ' + ' + '.join(m_features + [':'.join(p) for p in combinations(m_features, 2)])
        y_formula = 'y ~ ' + ' + '.join(y_features + [':'.join(p) for p in combinations(y_features, 2)])
    robjects.globalenv['df'] = df
    mediator_model = Rstats.lm(m_formula, data=base.as_symbol('df'))
    outcome_model = Rstats.lm(y_formula, data=base.as_symbol('df'))
    res = mediation.mediate(mediator_model, outcome_model, treat='t', mediator='m', boot=True, sims=1)

    relevant_variables = ['tau.coef', 'z1', 'z0', 'd1', 'd0']
    to_return = [np.array(res.rx2(v))[0] for v in relevant_variables]
    return to_return + [None]


def g_estimator(y, t, m, x):
    m = m.ravel()
    var_names = [[y, 'y'],
                 [t, 't'],
                 [m, 'm'],
                 [x, 'x']]
    df_list = list()
    for var, name in var_names:
        if len(var.shape) > 1:
            var_dim = var.shape[1]
            col_names = ['{}_{}'.format(name, i) for i in range(var_dim)]
            sub_df = pd.DataFrame(var, columns=col_names)
        else:
            sub_df = pd.DataFrame(var, columns=[name])
        df_list.append(sub_df)
        df = pd.concat(df_list, axis=1)
    m_features = [c for c in df.columns if ('x' in c)]
    y_features = [c for c in df.columns if ('x' in c)]
    t_features = [c for c in df.columns if ('x' in c)]
    m_formula = 'm ~ ' + ' + '.join(m_features)
    y_formula = 'y ~ ' + ' + '.join(y_features)
    t_formula = 't ~ ' + ' + '.join(t_features)
    robjects.globalenv['df'] = df
    res = plmed.G_estimation(t_formula,
                             m_formula,
                             y_formula,
                             exposure_family='binomial',
                             data=base.as_symbol('df'))
    direct_effect = res.rx2('coef')[0]
    indirect_effect = res.rx2('coef')[1]
    return [direct_effect + indirect_effect,
            direct_effect,
            direct_effect,
            indirect_effect,
            indirect_effect,
            None]


def bart(y, t, m, x, tmle=False):
    x_r, t_r, y_r = [_convert_array_to_R(uu) for uu in (x, t, y)]
    if tmle:
        bart_model = brc.bartc(y_r, t_r, x_r, method_rsp='tmle',
                               estimand='ate')
    else:
        bart_model = brc.bartc(y_r, t_r, x_r, method_rsp='p.weight',
                               estimand='ate')
    ate = brc.summary_bartcFit(bart_model).rx2('estimates')[0][0]
    return [ate] + [None] * 5


def medDML(y, t, m, x, trim=0.05, order=1):
    """
    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples, n_features_mediator)
            mediator value for each unit, can be continuous or binary, and
            multi-dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    trim    float
            Trimming rule for discarding observations with extreme
            conditional treatment or mediator probabilities
            (or products thereof). Observations with (products of)
            conditional probabilities that are smaller than trim in any
            denominator of the potential outcomes are dropped.
            Default is 0.05.

    order   integer
            If set to an integer larger than 1, then polynomials of that 
            order and interactions using the power series) rather than the 
            original control variables are used in the estimation of any
            conditional probability or conditional mean outcome. 
            Polynomials/interactions are created using the Generate.
            Powers command of the LARF package.
    """
    x_r, t_r, m_r, y_r = [base.as_matrix(_convert_array_to_R(uu)) for uu in (x, t, m, y)]
    res = causalweight.medDML(y_r, t_r, m_r, x_r, trim=trim, order=order)
    raw_res_R = np.array(res.rx2('results'))
    ntrimmed = res.rx2('ntrimmed')[0]
    return list(raw_res_R[0, :5]) + [ntrimmed]


def _convert_array_to_R(x):
    """
    converts a numpy array to a R matrix or vector
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.sum(a == np.array(_convert_array_to_R(a)))
    6
    """
    if len(x.shape) == 1:
        return robjects.FloatVector(x)
    elif len(x.shape) == 2:
        return robjects.r.matrix(robjects.FloatVector(x.ravel()),
                                 nrow=x.shape[0], byrow='TRUE')


def bootstrap_mediation(y, t, m, x, w=None, z=None, trim=0.0, logit=True,
                        boot=1999, estimator='huber_IPW_reg', seed=None):
    """
    master function, that runs a specific setting. A bit manual as some settings
    rely on calling a function, others on the parameters, and the signatures
    of the different functions are not 100% homogeneous.

    The estimator is computed once on the whole dataset, and then a user-defined
    number of bootstrap samples are used to re-fit the estimators, and get
    confidence intervals

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

    boot    integer, default 1999
            number of bootstrap samples

    estimator   string
                estimator and setting name

    seed    NoneType of integer
            seed to initiate the random state to draw the bootstrap samples

    returned values
    results     a table of size (3, 5), containing for the following estimates:
                total_effect, direct_effect_treated, direct_effect_control,
                indirect_effect_treated, indirect_effect_control
                on the first line the size of the effect, on the second line a
                bootstrap estimate of the standard deviation, and on the third
                line a pvalue of a statistical test (t-statistics, normal
                distribution) to assess whether the effect is different from zero
    n_samples   final number of samples used (in case of exclusions by trimming
                for example)

    """
    if estimator == "huber_IPW_R":
        x_r, t_r, m_r, y_r = [_convert_array_to_R(uu) for uu in (x, t, m, y)]
        output_w = causalweight.medweight(y=y_r, d=t_r, m=m_r, x=x_r, trim=trim,
                                          ATET="FALSE", logit="TRUE", boot=boot)
        raw_res_R = np.array(output_w.rx2('results'))
        return [raw_res_R]
    res_matrix = np.zeros((boot, 6))
    rg = default_rng(seed)
    if estimator == 'huber_IPW_reg':
        base_res = huber_IPW(y, t, m, x, w, z, trim, logit, penalty='l2', clip=0.01)
    elif estimator == 'huber_IPW_reg_NOclip':
        base_res = huber_IPW(y, t, m, x, w, z, trim, logit, penalty='l2', clip=0.0)
    elif estimator == 'huber_IPW_NOreg':
        base_res = huber_IPW(y, t, m, x, w, z, trim, logit, penalty='none', clip=0.01)
    elif estimator == 'huber_IPW_NOreg_NOclip':
        base_res = huber_IPW(y, t, m, x, w, z, trim, logit, penalty='none', clip=0.0)
    elif estimator == 'huber_IPW_RF':
        base_res = huber_IPW(y, t, m, x, w, z, trim, logit, forest=True)
    elif estimator == 'huber_IPW_RF_cross':
        base_res = huber_IPW(y, t, m, x, w, z, trim, logit, forest=True, crossfit=2)
    elif estimator == 'base_linear':
        base_res = ols_mediation(y, t, m, x, alpha_ridge=0.0)
    elif estimator == 'base_linear_ridge':
        base_res = ols_mediation(y, t, m, x, alpha_ridge=0.00001)
    elif estimator == 'alternative':
        base_res = alternative_estimator(y, t, m, x, alpha_ridge=0.0)
    elif estimator == 'alternative_ridge':
        base_res = alternative_estimator(y, t, m, x, alpha_ridge=0.00001)
    elif estimator == 'g-computation':
        base_res = g_computation(y, t, m, x)
    elif estimator == 'g-computation_inter':
        base_res = g_computation(y, t, m, x, interaction=True)
    elif estimator == 'g-computation_RF':
        base_res = g_computation(y, t, m, x, interaction=False, forest=True)
    elif estimator == 'g-computation_cross':
        base_res = g_computation(y, t, m, x, crossfit=2)
    elif estimator == 'g-computation_inter_cross':
        base_res = g_computation(y, t, m, x, interaction=True, crossfit=2)
    elif estimator == 'g-computation_RF_cross':
        base_res = g_computation(y, t, m, x, interaction=False, forest=True, crossfit=2)
    elif estimator == 'multiple':
        base_res = multiply_robust_efficient(y, t, m, x)
    elif estimator == 'multiple_inter':
        base_res = multiply_robust_efficient(y, t, m, x, interaction=True)
    elif estimator == 'multiple_RF':
        base_res = multiply_robust_efficient(y, t, m, x, interaction=False, forest=True)
    elif estimator == 'multiple_RF_cross':
        base_res = multiply_robust_efficient(y, t, m, x, interaction=False, forest=True, crossfit=2)
    elif estimator == 'R_mediate':
        base_res = r_mediate(y, t, m, x)
    elif estimator == 'R_mediate_inter':
        base_res = r_mediate(y, t, m, x, interaction=True)
    elif estimator == 'bart':
        base_res = bart(y, t, m, x, tmle=False)
    elif estimator == 'bart_tmle':
        base_res = bart(y, t, m, x, tmle=True)
    elif estimator == 'aipw_linear_cross':
        base_res = AIPW(y, t, m, x, crossfit=2)
    elif estimator == 'aipw_linear_cross_noreg':
        base_res = AIPW(y, t, m, x, crossfit=2, clip=0, alpha_ridge=0)
    elif estimator == 'aipw_RF':
        base_res = AIPW(y, t, m, x, forest=True)
    elif estimator == 'aipw_RF_cross':
        base_res = AIPW(y, t, m, x, forest=True, crossfit=2)
    elif estimator == 'aipw_forest_R':
        base_res = AIPW(y, t, m, x, forest_r=True)
    elif estimator == 'medDML_trim':
        base_res = medDML(y, t, m, x, trim=0.05, order=1)
    elif estimator == 'medDML_NOtrim':
        base_res = medDML(y, t, m, x, trim=0.0, order=1)
    elif estimator == 'medDML_trim_inter':
        base_res = medDML(y, t, m, x, trim=0.05, order=2)
    elif estimator == 'g_estimator':
        base_res = g_estimator(y, t, m, x)
    else:
        raise ValueError('estimator should be one of ["huber_IPW", "base_linear", "alternative", "g-computation", "g-computation_inter"]')
    for i in range(boot):
        ind = rg.choice(len(y), len(y), replace=True)
        y_b, t_b, m_b, x_b = y[ind], t[ind], m[ind], x[ind]
        if w is not None:
            w_b = w[ind]
        else:
            w_b = None
        if z is not None:
            z_b = z[ind]
        else:
            z_b = None
        if estimator == 'huber_IPW_reg':
            res_matrix[i, :] = huber_IPW(y_b, t_b, m_b, x_b, w_b, z_b,
                                         trim, logit, penalty='l2', clip=0.01)
        elif estimator == 'huber_IPW_reg_NOclip':
            res_matrix[i, :] = huber_IPW(y_b, t_b, m_b, x_b, w_b, z_b, trim,
                                         logit, penalty='l2', clip=0.0)
        elif estimator == 'huber_IPW_NOreg':
            res_matrix[i, :] = huber_IPW(y_b, t_b, m_b, x_b, w_b, z_b,
                                         trim, logit, penalty='none', clip=0.01)
        elif estimator == 'huber_IPW_NOreg_NOclip':
            res_matrix[i, :] = huber_IPW(y_b, t_b, m_b, x_b, w_b, z_b, trim,
                                         logit, penalty='none', clip=0.0)
        elif estimator == 'huber_IPW_RF':
            res_matrix[i, :] = huber_IPW(y_b, t_b, m_b, x_b, w_b, z_b,
                                         trim, logit, forest=True)
        elif estimator == 'huber_IPW_RF_cross':
            res_matrix[i, :] = huber_IPW(y_b, t_b, m_b, x_b, w_b, z_b,
                                         trim, logit, forest=True, crossfit=2)
        elif estimator == 'base_linear':
            res_matrix[i, :] = ols_mediation(y_b, t_b, m_b, x_b, alpha_ridge=0.0)
        elif estimator == 'base_linear_ridge':
            res_matrix[i, :] = ols_mediation(y_b, t_b, m_b, x_b, alpha_ridge=0.00001)
        elif estimator == 'alternative':
            res_matrix[i, :] = alternative_estimator(y_b, t_b, m_b, x_b, alpha_ridge=0.0)
        elif estimator == 'alternative_ridge':
            res_matrix[i, :] = alternative_estimator(y_b, t_b, m_b, x_b, alpha_ridge=0.00001)
        elif estimator == 'g-computation':
            res_matrix[i, :] = g_computation(y_b, t_b, m_b, x_b)
        elif estimator == 'g-computation_inter':
            res_matrix[i, :] = g_computation(y_b, t_b, m_b, x_b, interaction=True)
        elif estimator == 'g-computation_RF':
            res_matrix[i, :] = g_computation(y_b, t_b, m_b, x_b, interaction=False, forest=True)
        elif estimator == 'g-computation_cross':
            res_matrix[i, :] = g_computation(y_b, t_b, m_b, x_b, crossfit=2)
        elif estimator == 'g-computation_inter_cross':
            res_matrix[i, :] = g_computation(y_b, t_b, m_b, x_b, interaction=True, crossfit=2)
        elif estimator == 'g-computation_RF_cross':
            res_matrix[i, :] = g_computation(y_b, t_b, m_b, x_b, interaction=False, forest=True, crossfit=2)
        elif estimator == 'multiple':
            res_matrix[i, :] = multiply_robust_efficient(y_b, t_b, m_b, x_b)
        elif estimator == 'multiple_inter':
            res_matrix[i, :] = multiply_robust_efficient(y_b, t_b, m_b, x_b, interaction=True)
        elif estimator == 'multiple_RF':
            res_matrix[i, :] = multiply_robust_efficient(y_b, t_b, m_b, x_b, interaction=False, forest=True)
        elif estimator == 'multiple_RF_cross':
            res_matrix[i, :] = multiply_robust_efficient(y_b, t_b, m_b, x_b, interaction=False, forest=True, crossfit=2)
        elif estimator == 'R_mediate':
            res_matrix[i, :] = r_mediate(y_b, t_b, m_b, x_b)
        elif estimator == 'R_mediate_inter':
            res_matrix[i, :] = r_mediate(y_b, t_b, m_b, x_b, interaction=True)
        elif estimator == 'bart':
            res_matrix[i, :] = bart(y_b, t_b, m_b, x_b, tmle=False)
        elif estimator == 'bart_tmle':
            res_matrix[i, :] = bart(y_b, t_b, m_b, x_b, tmle=True)
        elif estimator == 'aipw_linear_cross':
            res_matrix[i, :] = AIPW(y_b, t_b, m_b, x_b, crossfit=2)
        elif estimator == 'aipw_linear_cross_noreg':
            res_matrix[i, :] = AIPW(y_b, t_b, m_b, x_b, crossfit=2, clip=0, alpha_ridge=0)
        elif estimator == 'aipw_RF':
            res_matrix[i, :] = AIPW(y_b, t_b, m_b, x_b, forest=True)
        elif estimator == 'aipw_RF_cross':
            res_matrix[i, :] = AIPW(y_b, t_b, m_b, x_b, forest=True, crossfit=2)
        elif estimator == 'aipw_forest_R':
            res_matrix[i, :] = AIPW(y_b, t_b, m_b, x_b, forest_r=True)
        elif res_matrix[i, :] == 'medDML_trim':
            res_matrix[i, :] = medDML(y_b, t_b, m_b, x_b, trim=0.05, order=1)
        elif res_matrix[i, :] == 'medDML_NOtrim':
            res_matrix[i, :] = medDML(y_b, t_b, m_b, x_b, trim=0.0, order=1)
        elif res_matrix[i, :] == 'medDML_trim_inter':
            res_matrix[i, :] = medDML(y_b, t_b, m_b, x_b, trim=0.05, order=2)
        elif estimator == 'g_estimator':
            res_matrix[i, :] = g_estimator(y_b, t_b, m_b, x_b)
        else:
            raise ValueError('estimator should be one of ["huber_IPW", "base_linear", "alternative", "g-computation", "g-computation_inter"]')
    val_effect = np.array(base_res[:-1])
    std_effect = np.std(res_matrix[:, :-1], axis=0, ddof=1)
    try:
        pval_effect = 2 * stats.norm.cdf(-abs(val_effect / std_effect))
    except:
        pval_effect = np.nan * np.ones(val_effect.shape)
    return np.vstack((val_effect, std_effect, pval_effect)), base_res[-1]


def logit_rec(x):
    """
    reciprocal function of the logit
    """
    return np.exp(x) / (1 + np.exp(x))


def transform_x(x, severe=False, seed=None, binary=True):
    rg_transform = rg_coef = default_rng(seed)
    binary_function_list = ['xor', 'sum', 'product', 'exp', 'log', 'quotient']
    if binary:
        function_list = list(binary_function_list)
    else:
        function_list = list(binary_function_list[1:])

    if severe:
        n, p = x.shape
        if p == 1:
            return np.exp(x)
        new_x = np.zeros((n, p))
        for i in range(p):
            function = rg_transform.choice(function_list)
            if function == "xor":
                x1, x2 = rg_transform.choice(p, 2, replace=False)
                new_x[:, i] = np.logical_xor(x[:, x1], x[:, x2]).astype(int)
            elif function == "sum":
                ns = rg_transform.choice(p-1) + 1
                indexes = rg_transform.choice(p, ns, replace=False)
                new_x[:, i] = np.sum(x[:, indexes], axis=1)
            elif function == "product":
                ns = rg_transform.choice(p-1) + 1
                indexes = rg_transform.choice(p, ns, replace=False)
                new_x[:, i] = np.prod(x[:, indexes], axis=1)
            elif function == "exp":
                x1 = rg_transform.choice(p)
                new_x[:, i] = np.exp(x[:, x1])
            elif function == "log":
                x1 = rg_transform.choice(p)
                new_x[:, i] = np.log(x[:, x1] + 1)
            elif function == "quotient":
                x1, x2 = rg_transform.choice(p, 2, replace=False)
                new_x[:, i] = x[:, x1] / (1 + x[:, x2])
        return new_x.copy()
        # to change - return transforms like xor, product or quotient
        # between variables etc
    else:
        return x.copy()


def simulate_data_Mbin_Ycont(n, rg, dim_x_observed=1,
                             interaction_xt_m=False, interaction_xt_y=False,
                             interaction_xm_y=False, interaction_tm_y=False,
                             misspecification_m=False,
                             misspecification_y=False,
                             severe_misspecification_m=False,
                             severe_misspecification_y=False,
                             seed=None,
                             x_gen='normal', p_x=None, dim_x=1):
    """
    this function simulates data under a number of different scenarios

    n                   integer
                        number of observations in the simulated dataset

    rg                  random generator (numpy object class Generator)
                        used to generate the data

    dim_x_observed      integer
                        number of dimnsions of the potential confounder
                        covariates

    interaction_xt_m    boolean, default False
                        interaction between confounders (X) and treatment (T)
                        in the model for the mediator (M)
                        (term XT in an otherwise linear model)

    interaction_xt_y    boolean, default False
                        interaction between confounders (X) and treatment (T)
                        in the model for the outcome (Y)
                        (term XT in an otherwise linear model)

    interaction_xm_y    boolean, default False
                        interaction between confounders (X) and mediator (M)
                        in the model for the outcome (Y)
                        (term XM in an otherwise linear model)

    interaction_tm_y    boolean, default False
                        interaction between the mediator (M) and treatment (T)
                        in the model for the outcome (Y)
                        (term TM in an otherwise linear model)

    misspecification_m  boolean, default False
                        nonlinearity in the model for the mediator (M)
                        consisting in adding a quadratic term for the
                        confounder (X) in an otherwise linear model

    misspecification_y  boolean, default False
                        nonlinearity in the model for the outcome (Y)
                        consisting in adding a quadratic term for the
                        confounder (X) in an otherwise linear model

    severe_misspecification_m  boolean, default False
                               nonlinearity in the model for the mediator (M)
                               consisting in adding a quadratic term for the
                               confounder (X) in an otherwise linear model
                               nonlinearity is more severe and includes log,
                               exp, xor between the dimensions of X. If X is
                               of dimension one, exp is selected, otherwise
                               a different operation is drawn to provide a new
                               variable X with the same dimension as the input.

    severe_misspecification_y  boolean, default False
                               nonlinearity in the model for the outcome (Y)
                               consisting in adding a quadratic term for the
                               confounder (X) in an otherwise linear model
                               nonlinearity is more severe and includes log,
                               exp, xor between the dimensions of X. If X is
                               of dimension one, exp is selected, otherwise
                               a different operation is drawn to provide a new
                               variable X with the same dimension as the input.

    seed                integer, default is None
                        seed for the generation of model coefficients
                        allows to generate several datasets with the same
                        coefficients

    x_gen               string among "normal", "binary"
                        nature of confounder variables X, either from a normal
                        distribution, or a benoulli distribution, with
                        parameter p_x, that is either provided as input to
                        the function, or drawn from a uniform distribution
                        on the interval [0, 1[

    p_x                 array of size dim_x_observed, default None
                        parameter of the bernoulli distributions to draw a
                        binary confounder variable

    dim_x               integer
                        true number of confounder dimensions. Only the first
                        dim_x dimensions of X actually go into the model.

    returns vectors or arrays y, t, m, x
    and two theoretical values for the 5 effects of interest
    (i) obtained by simulation, i.e. getting the same random effect, and get
        the variables under treatment or not
    (ii) theoretical values computed using the coefficients (and some of the
         data, in particular covariates and treatment).
    """
    rg_coef = default_rng(seed)
    bias = np.ones((n, 1))
    poly = PolynomialFeatures(degree=2,
                              interaction_only=False,
                              include_bias=True)

    if x_gen == 'normal':
        x_observed = rg.standard_normal(n * dim_x_observed)\
                       .reshape((n, dim_x_observed))
    elif x_gen == 'binary':
        if p_x is None:
            p_x = rg.random(dim_x_observed)
        x_observed = np.zeros((n, dim_x_observed))
        for k in range(dim_x_observed):
            x_observed[:, k] = rg.binomial(1, p_x[k], size=n)
    x = x_observed[:, :dim_x]

    poly_x = np.hstack((bias, x))
    alphas = rg_coef.standard_normal(poly_x.shape[1])
    p_t = np.exp(alphas.dot(poly_x.T)) /\
        (1 + np.exp(alphas.dot(poly_x.T)))
    t = rg.binomial(1, p_t, n).reshape(-1, 1)
    t0 = np.zeros((n, 1))
    t1 = np.ones((n, 1))

    x_m = transform_x(x, severe=severe_misspecification_m)
    pre_xt = poly.fit_transform(x_m)
    betas_0x = rg_coef.standard_normal(dim_x + 1)
    if misspecification_m:
        betas_x2 = rg_coef.standard_normal(int(dim_x * (dim_x + 1) / 2)) * 4
    else:
        betas_x2 = np.zeros(int(dim_x * (dim_x + 1) / 2))
    t_rep = np.repeat(t, dim_x, axis=1)
    poly_xt = np.hstack((pre_xt, x_m * t_rep, t))
    t0_rep = np.repeat(t0, dim_x, axis=1)
    poly_xt0 = np.hstack((pre_xt, x_m * t0_rep, t0))
    t1_rep = np.repeat(t1, dim_x, axis=1)
    poly_xt1 = np.hstack((pre_xt, x_m * t1_rep, t1))
    if interaction_xt_m:
        betas_xt = rg_coef.standard_normal(dim_x) * 4
    else:
        betas_xt = np.zeros(dim_x)
    betas_t = rg_coef.standard_normal(1) + 0.5
    betas = np.hstack((betas_0x, betas_x2, betas_xt, betas_t))

    p_m0 = logit_rec(betas.dot(poly_xt0.T))
    p_m1 = logit_rec(betas.dot(poly_xt1.T))
    pre_m = rg.random(n)
    m0 = ((pre_m < p_m0) * 1).reshape(-1, 1)
    m1 = ((pre_m < p_m1) * 1).reshape(-1, 1)
    m_2d = np.hstack((m0, m1))
    m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)

    x_y = transform_x(x, severe=severe_misspecification_y)
    pre_xtm = poly.fit_transform(x_y)
    gammas_0x = rg_coef.standard_normal(dim_x + 1)
    if misspecification_y:
        gammas_x2 = rg_coef.standard_normal(int(dim_x * (dim_x + 1) / 2)) * 4
    else:
        gammas_x2 = np.zeros(int(dim_x * (dim_x + 1) / 2))
    t_rep = np.repeat(t, dim_x, axis=1)
    xt = x_y * t_rep
    t0_rep = np.repeat(t0, dim_x, axis=1)
    xt0 = x_y * t0_rep
    t1_rep = np.repeat(t1, dim_x, axis=1)
    xt1 = x_y * t1_rep
    if interaction_xt_y:
        gammas_xt = rg_coef.standard_normal(dim_x) * 4
    else:
        gammas_xt = np.zeros(dim_x)

    m_rep = np.repeat(m, dim_x, axis=1)
    xm = x_y * m_rep
    m0_rep = np.repeat(m0, dim_x, axis=1)
    xm0 = x_y * m0_rep
    m1_rep = np.repeat(m1, dim_x, axis=1)
    xm1 = x_y * m1_rep
    if interaction_xm_y:
        gammas_xm = rg_coef.standard_normal(dim_x) * 4
    else:
        gammas_xm = np.zeros(dim_x)
    tm = t * m
    t0m0 = t0 * m0
    t1m1 = t1 * m1
    t0m1 = t0 * m1
    t1m0 = t1 * m0
    if interaction_tm_y:
        gammas_tm = rg_coef.standard_normal(1) * 4
    else:
        gammas_tm = np.zeros(1)
    gammas_t = np.array([2.5])
    gammas_m = np.array([3])

    xtm = np.hstack((pre_xtm, xt, xm, tm, t, m))
    xt0m0 = np.hstack((pre_xtm, xt0, xm0, t0m0, t0, m0))
    xt1m1 = np.hstack((pre_xtm, xt1, xm1, t1m1, t1, m1))
    xt0m1 = np.hstack((pre_xtm, xt0, xm1, t0m1, t0, m1))
    xt1m0 = np.hstack((pre_xtm, xt1, xm0, t1m0, t1, m0))
    gammas = np.hstack((gammas_0x, gammas_x2, gammas_xt, gammas_xm, gammas_tm,
                        gammas_t, gammas_m))
    Uy = rg.standard_normal(n)
    y = gammas.dot(xtm.T) + Uy

    y1m1 = gammas.dot(xt1m1.T) + Uy
    y0m0 = gammas.dot(xt0m0.T) + Uy
    y0m1 = gammas.dot(xt0m1.T) + Uy
    y1m0 = gammas.dot(xt1m0.T) + Uy

    total_effect = (y1m1 - y0m0).sum() / n
    direct_effect_control = (y1m0 - y0m0).sum() / n
    direct_effect_treated = (y1m1 - y0m1).sum() / n
    indirect_effect_control = (y0m1 - y0m0).sum() / n
    indirect_effect_treated = (y1m1 - y1m0).sum() / n
    empirical_truth = (total_effect,
                       direct_effect_treated,
                       direct_effect_control,
                       indirect_effect_treated,
                       indirect_effect_control)

    # compute theoretical effect values
    direct_effect_control = (gammas_t + gammas_tm * p_m0).mean()
    direct_effect_treated = (gammas_t + gammas_tm * p_m1).mean()
    indirect_effect_control = ((p_m1 - p_m0) * (gammas_m + x_y.dot(gammas_xm))).mean()
    indirect_effect_treated = ((p_m1 - p_m0) * (gammas_m + x_y.dot(gammas_xm) + gammas_tm)).mean()
    total_effect = direct_effect_control + indirect_effect_treated
    theoretical_truth = (total_effect,
                         direct_effect_treated,
                         direct_effect_control,
                         indirect_effect_treated,
                         indirect_effect_control)

    return y, t[:, 0], m[:, 0], x_observed, empirical_truth, theoretical_truth, alphas, betas, gammas
