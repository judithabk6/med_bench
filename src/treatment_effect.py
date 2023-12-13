"""
the objective of this script is to implement estimators without mediation in
causal inference, simulate data, and evaluate and compare estimators
"""

# using rpy2 to have the same data in R and python...

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri, numpy2ri
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

pandas2ri.activate()
numpy2ri.activate()

causalweight = rpackages.importr('causalweight')
mediation = rpackages.importr('mediation')
Rstats = rpackages.importr('stats')
base = rpackages.importr('base')
grf = rpackages.importr('grf')
plmed = rpackages.importr('plmed')

ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5
TINY = 1.e-12


def plain_IPW(y, t, x, trim=0.01, regularization=True):
    """
    plain IPW estimator without mediation
    """
    if regularization:
        cs = ALPHAS
    else:
        cs = [np.inf]
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    p_x_clf = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS).fit(x, t)
    p_x = p_x_clf.predict_proba(x)[:, 1]
    # trimming
    p_x[p_x < trim] = trim
    p_x[p_x > 1 - trim] = 1 - trim
    y1m1 = np.sum(y * t / p_x) / np.sum(t / p_x)
    y0m0 = np.sum(y * (1 - t) / (1 - p_x)) /\
        np.sum((1 - t) / (1 - p_x))
    return y1m1 - y0m0


def AIPW(y, t, m, x, clip=0.01, forest=False, crossfit=0, forest_r=False,
         regularization=True):
    """
    AIPW estimator
    """
    if regularization:
        alphas = ALPHAS
        cs = ALPHAS
    else:
        alphas = [TINY]
        cs = [np.inf]
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
                y_reg_treated = RidgeCV(alphas=alphas, cv=CV_FOLDS)\
                    .fit(x[treated_train_index, :], y[treated_train_index])
                y_reg_control = RidgeCV(alphas=alphas, cv=CV_FOLDS)\
                    .fit(x[control_train_index, :], y[control_train_index])
                t_prob = LogisticRegressionCV(Cs=cs, cv=CV_FOLDS)\
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