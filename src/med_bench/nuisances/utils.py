"""
the objective of this script is to provide nuisance estimators 
for mediation in causal inference
"""

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV

from med_bench.utils.constants import ALPHAS, CV_FOLDS, TINY


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
        clf = LogisticRegressionCV(random_state=random_state, Cs=cs, cv=CV_FOLDS)
    else:
        clf = RandomForestClassifier(
            random_state=random_state, n_estimators=100, min_samples_leaf=10
        )
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
