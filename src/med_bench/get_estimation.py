#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

from .mediation import (
    mediation_IPW,
    mediation_coefficient_product,
    mediation_g_formula,
    mediation_multiply_robust,
    mediation_dml,
    r_mediation_g_estimator,
    r_mediation_dml,
    r_mediate,
)

from med_bench.estimation.mediation_coefficient_product import CoefficientProduct
from med_bench.estimation.mediation_dml import DoubleMachineLearning
from med_bench.estimation.mediation_g_computation import GComputation
from med_bench.estimation.mediation_ipw import InversePropensityWeighting
from med_bench.estimation.mediation_mr import MultiplyRobust
from med_bench.nuisances.utils import _get_regularization_parameters
from med_bench.utils.constants import CV_FOLDS

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.calibration import CalibratedClassifierCV


def transform_outputs(causal_effects):
    """Transforms outputs in the old format

    Args:
        causal_effects (dict): dictionary of causal effects

    Returns:
        list: list of causal effects
    """
    total = causal_effects['total_effect']
    direct_treated = causal_effects['direct_effect_treated']
    direct_control = causal_effects['direct_effect_control']
    indirect_treated = causal_effects['indirect_effect_treated']
    indirect_control = causal_effects['indirect_effect_control']
    return [total, direct_treated, direct_control, indirect_treated, indirect_control, 0]


def get_estimation(x, t, m, y, estimator, config):
    """Wrapper estimator fonction ; calls an estimator given mediation data
    in order to estimate total, direct, and indirect effects.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features_covariates)
        Covariates value for each unit
    t : array-like, shape (n_samples)
        Treatment value for each unit
    m : array-like, shape (n_samples, n_features_mediator)
        Mediator value for each unit
    y : array-like, shape (n_samples)
        Outcome value for each unit
    estimator : str
        Label of the estimator
    config : int
        Indicates whether the estimator is suited to the data.
        Should be 1 if dim_m=1 and type_m="binary", 5 otherwise.
        This is a legacy parameter, will be removed in future updates.

    Returns
    -------
    list
        A list of estimated effects :
        [total effect,
        direct effect on the exposed,
        direct effect on the unexposed,
        indirect effect on the exposed,
        indirect effect on the unexposed,
        number of discarded samples OR non-discarded samples]

    Raises
    ------
    UserWarning
        If estimator name is misspelled.
    """
    effects = None
    if estimator == "mediation_IPW_R":
        x_r, t_r, m_r, y_r = [_convert_array_to_R(uu) for uu in (x, t, m, y)]
        output_w = causalweight.medweight(
            y=y_r, d=t_r, m=m_r, x=x_r, trim=0.0, ATET="FALSE", logit="TRUE", boot=2
        )
        raw_res_R = np.array(output_w.rx2("results"))
        effects = raw_res_R[0, :]
    elif estimator == "coefficient_product":
        effects = mediation_coefficient_product(y, t, m, x)
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator = CoefficientProduct(
            regressor=reg, classifier=clf, regularize=True)
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_noreg":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=False,
            forest=False,
            crossfit=0,
            clip=1e-6,
            calibration=None,
        )
        cs, alphas = _get_regularization_parameters(regularization=False)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        estimator = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=clf
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_ipw_noreg_cf":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=False,
            forest=False,
            crossfit=2,
            clip=1e-6,
            calibration=None,
        )
    elif estimator == "mediation_ipw_reg":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=False,
            crossfit=0,
            clip=1e-6,
            calibration=None,
        )
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        estimator = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=clf
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_ipw_reg_cf":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=False,
            crossfit=2,
            clip=1e-6,
            calibration=None,
        )
    elif estimator == "mediation_ipw_reg_calibration":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=False,
            crossfit=0,
            clip=1e-6,
            calibration=None,
        )
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        estimator = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=CalibratedClassifierCV(clf, method="sigmoid")
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_ipw_reg_calibration_iso":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=False,
            crossfit=0,
            clip=1e-6,
            calibration="isotonic",
        )
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        estimator = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=CalibratedClassifierCV(clf, method="isotonic")
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_ipw_reg_calibration_cf":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=False,
            crossfit=2,
            clip=1e-6,
            calibration='sigmoid',
        )
    elif estimator == "mediation_ipw_reg_calibration_iso_cf":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=False,
            crossfit=2,
            clip=1e-6,
            calibration="isotonic",
        )
    elif estimator == "mediation_ipw_forest":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=True,
            crossfit=0,
            clip=1e-6,
            calibration=None,
        )
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=clf
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_ipw_forest_cf":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=True,
            crossfit=2,
            clip=1e-6,
            calibration=None,
        )
    elif estimator == "mediation_ipw_forest_calibration":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=True,
            crossfit=0,
            clip=1e-6,
            calibration=None,
        )
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=CalibratedClassifierCV(clf, method="sigmoid")
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_ipw_forest_calibration_iso":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=True,
            crossfit=0,
            clip=1e-6,
            calibration="isotonic",
        )
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=CalibratedClassifierCV(clf, method="isotonic")
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_ipw_forest_calibration_cf":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=True,
            crossfit=2,
            clip=1e-6,
            calibration='sigmoid',
        )
    elif estimator == "mediation_ipw_forest_calibration_iso_cf":
        effects = mediation_IPW(
            y,
            t,
            m,
            x,
            trim=0,
            regularization=True,
            forest=True,
            crossfit=2,
            clip=1e-6,
            calibration="isotonic",
        )
    elif estimator == "mediation_g_computation_noreg":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=False,
                crossfit=0,
                regularization=False,
                calibration=None,
            )
            cs, alphas = _get_regularization_parameters(regularization=False)
            clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
            reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
            estimator = GComputation(regressor=reg, classifier=clf)
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_g_computation_noreg_cf":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=False,
                crossfit=2,
                regularization=False,
                calibration=None,
            )
    elif estimator == "mediation_g_computation_reg":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=False,
                crossfit=0,
                regularization=True,
                calibration=None,
            )
            cs, alphas = _get_regularization_parameters(regularization=True)
            clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
            reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
            estimator = GComputation(regressor=reg, classifier=clf)
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_g_computation_reg_cf":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=False,
                crossfit=2,
                regularization=True,
                calibration=None,
            )
    elif estimator == "mediation_g_computation_reg_calibration":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=False,
                crossfit=0,
                regularization=True,
                calibration=None,
            )
            cs, alphas = _get_regularization_parameters(regularization=True)
            clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
            reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
            estimator = GComputation(
                regressor=reg, classifier=CalibratedClassifierCV(clf, method="sigmoid"))
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_g_computation_reg_calibration_iso":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=False,
                crossfit=0,
                regularization=True,
                calibration="isotonic",
            )
            cs, alphas = _get_regularization_parameters(regularization=True)
            clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
            reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
            estimator = GComputation(
                regressor=reg, classifier=CalibratedClassifierCV(clf, method="isotonic"))
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_g_computation_reg_calibration_cf":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=False,
                crossfit=2,
                regularization=True,
                calibration='sigmoid',
            )
    elif estimator == "mediation_g_computation_reg_calibration_iso_cf":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=False,
                crossfit=2,
                regularization=True,
                calibration="isotonic",
            )
    elif estimator == "mediation_g_computation_forest":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=True,
                crossfit=0,
                regularization=True,
                calibration=None,
            )
            cs, alphas = _get_regularization_parameters(regularization=True)
            clf = RandomForestClassifier(
                random_state=42, n_estimators=100, min_samples_leaf=10)
            reg = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=10, random_state=42)
            estimator = GComputation(regressor=reg, classifier=clf)
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_g_computation_forest_cf":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=True,
                crossfit=2,
                regularization=True,
                calibration=None,
            )
    elif estimator == "mediation_g_computation_forest_calibration":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=True,
                crossfit=0,
                regularization=True,
                calibration='sigmoid',
            )
            cs, alphas = _get_regularization_parameters(regularization=True)
            clf = RandomForestClassifier(
                random_state=42, n_estimators=100, min_samples_leaf=10)
            reg = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=10, random_state=42)
            estimator = GComputation(
                regressor=reg, classifier=CalibratedClassifierCV(clf, method="sigmoid"))
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_forest_calibration_iso":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=True,
                crossfit=0,
                regularization=True,
                calibration="isotonic",
            )
            cs, alphas = _get_regularization_parameters(regularization=True)
            clf = RandomForestClassifier(
                random_state=42, n_estimators=100, min_samples_leaf=10)
            reg = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=10, random_state=42)
            estimator = GComputation(
                regressor=reg, classifier=CalibratedClassifierCV(clf, method="isotonic"))
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_forest_calibration_cf":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=True,
                crossfit=2,
                regularization=True,
                calibration='sigmoid',
            )
    elif estimator == "mediation_g_computation_forest_calibration_iso_cf":
        if config in (0, 1, 2):
            effects = mediation_g_formula(
                y,
                t,
                m,
                x,
                interaction=False,
                forest=True,
                crossfit=2,
                regularization=True,
                calibration="isotonic",
            )
    elif estimator == "mediation_multiply_robust_noreg":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=False,
                crossfit=0,
                clip=1e-6,
                regularization=False,
                calibration=None,
            )
            cs, alphas = _get_regularization_parameters(regularization=False)
            clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
            reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
            estimator = MultiplyRobust(
                clip=1e-6, ratio="propensities", normalized=True, regressor=reg,
                classifier=clf)
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)

    elif estimator == "mediation_multiply_robust_noreg_cf":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=False,
                crossfit=2,
                clip=1e-6,
                regularization=False,
                calibration=None,
            )
    elif estimator == "mediation_multiply_robust_reg":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=False,
                crossfit=0,
                clip=1e-6,
                regularization=True,
                calibration=None,
            )
            cs, alphas = _get_regularization_parameters(regularization=True)
            clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
            reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
            estimator = MultiplyRobust(
                clip=1e-6, ratio="propensities", normalized=True, regressor=reg,
                classifier=clf)
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_multiply_robust_reg_cf":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=False,
                crossfit=2,
                clip=1e-6,
                regularization=True,
                calibration=None,
            )
    elif estimator == "mediation_multiply_robust_reg_calibration":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=False,
                crossfit=0,
                clip=1e-6,
                regularization=True,
                calibration='sigmoid',
            )
            cs, alphas = _get_regularization_parameters(regularization=False)
            clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
            reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
            estimator = MultiplyRobust(
                clip=1e-6, ratio="propensities", normalized=True, regressor=reg,
                classifier=CalibratedClassifierCV(clf, method="sigmoid"))
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_multiply_robust_reg_calibration_iso":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=False,
                crossfit=0,
                clip=1e-6,
                regularization=True,
                calibration="isotonic",
            )
            cs, alphas = _get_regularization_parameters(regularization=False)
            clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
            reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
            estimator = MultiplyRobust(
                clip=1e-6, ratio="propensities", normalized=True, regressor=reg,
                classifier=CalibratedClassifierCV(clf, method="isotonic"))
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_multiply_robust_reg_calibration_cf":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=False,
                crossfit=2,
                clip=1e-6,
                regularization=True,
                calibration='sigmoid',
            )
    elif estimator == "mediation_multiply_robust_reg_calibration_iso_cf":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=False,
                crossfit=2,
                clip=1e-6,
                regularization=True,
                calibration="isotonic",
            )
    elif estimator == "mediation_multiply_robust_forest":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=True,
                crossfit=0,
                clip=1e-6,
                regularization=True,
                calibration=None,
            )
            cs, alphas = _get_regularization_parameters(regularization=False)
            clf = RandomForestClassifier(
                random_state=42, n_estimators=100, min_samples_leaf=10)
            reg = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=10, random_state=42)
            estimator = MultiplyRobust(
                clip=1e-6, ratio="propensities", normalized=True, regressor=reg,
                classifier=clf)
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_multiply_robust_forest_cf":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=True,
                crossfit=2,
                clip=1e-6,
                regularization=True,
                calibration=None,
            )
    elif estimator == "mediation_multiply_robust_forest_calibration":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=True,
                crossfit=0,
                clip=1e-6,
                regularization=True,
                calibration='sigmoid',
            )
            cs, alphas = _get_regularization_parameters(regularization=False)
            clf = RandomForestClassifier(
                random_state=42, n_estimators=100, min_samples_leaf=10)
            reg = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=10, random_state=42)
            estimator = MultiplyRobust(
                clip=1e-6, ratio="propensities", normalized=True, regressor=reg,
                classifier=CalibratedClassifierCV(clf, method="sigmoid"))
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_multiply_robust_forest_calibration_iso":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=True,
                crossfit=0,
                clip=1e-6,
                regularization=True,
                calibration="isotonic",
            )
            cs, alphas = _get_regularization_parameters(regularization=False)
            clf = RandomForestClassifier(
                random_state=42, n_estimators=100, min_samples_leaf=10)
            reg = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=10, random_state=42)
            estimator = MultiplyRobust(
                clip=1e-6, ratio="propensities", normalized=True, regressor=reg,
                classifier=CalibratedClassifierCV(clf, method="isotonic"))
            estimator.fit(t, m, x, y)
            causal_effects = estimator.estimate(t, m, x, y)
            effects = transform_outputs(causal_effects)
    elif estimator == "mediation_multiply_robust_forest_calibration_cf":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=True,
                crossfit=2,
                clip=1e-6,
                regularization=True,
                calibration='sigmoid',
            )
    elif estimator == "mediation_multiply_robust_forest_calibration_iso_cf":
        if config in (0, 1, 2):
            effects = mediation_multiply_robust(
                y,
                t,
                m.astype(int),
                x,
                interaction=False,
                forest=True,
                crossfit=2,
                clip=1e-6,
                regularization=True,
                calibration="isotonic",
            )
    elif estimator == "simulation_based":
        if config in (0, 1, 2):
            effects = r_mediate(y, t, m, x, interaction=False)
    elif estimator == "mediation_dml":
        if config > 0:
            effects = r_mediation_dml(y, t, m, x, trim=0.0, order=1)
    elif estimator == "mediation_dml_noreg":
        effects = mediation_dml(
            y,
            t,
            m,
            x,
            trim=0,
            clip=1e-6,
            regularization=False,
            calibration=None)
        cs, alphas = _get_regularization_parameters(regularization=False)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        estimator = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=clf
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_dml_reg":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, calibration=None)
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        estimator = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=clf
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_dml_reg_fixed_seed":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, random_state=321, calibration=None)
    elif estimator == "mediation_dml_noreg_cf":
        effects = mediation_dml(
            y,
            t,
            m,
            x,
            trim=0,
            clip=1e-6,
            crossfit=2,
            regularization=False,
            calibration=None)
    elif estimator == "mediation_dml_reg_cf":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, crossfit=2, calibration=None)
    elif estimator == "mediation_dml_reg_calibration":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, crossfit=0, calibration='sigmoid')
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        estimator = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=CalibratedClassifierCV(clf, method="sigmoid")
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_dml_forest":
        effects = mediation_dml(
            y,
            t,
            m,
            x,
            trim=0,
            clip=1e-6,
            crossfit=0,
            calibration=None,
            forest=True)
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=clf
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_dml_forest_calibration":
        effects = mediation_dml(
            y,
            t,
            m,
            x,
            trim=0,
            clip=1e-6,
            crossfit=0,
            calibration='sigmoid',
            forest=True)
        cs, alphas = _get_regularization_parameters(regularization=True)
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=CalibratedClassifierCV(clf, method="sigmoid")
        )
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)
    elif estimator == "mediation_dml_reg_calibration_cf":
        effects = mediation_dml(
            y,
            t,
            m,
            x,
            trim=0,
            clip=1e-6,
            crossfit=2,
            calibration='sigmoid',
            forest=False)
    elif estimator == "mediation_dml_forest_cf":
        effects = mediation_dml(
            y,
            t,
            m,
            x,
            trim=0,
            clip=1e-6,
            crossfit=2,
            calibration=None,
            forest=True)
    elif estimator == "mediation_dml_forest_calibration_cf":
        effects = mediation_dml(
            y,
            t,
            m,
            x,
            trim=0,
            clip=1e-6,
            crossfit=2,
            calibration='sigmoid',
            forest=True)
    elif estimator == "mediation_g_estimator":
        if config in (0, 1, 2):
            effects = r_mediation_g_estimator(y, t, m, x)
    else:
        raise ValueError("Unrecognized estimator label.")
    if effects is None:
        if config not in (0, 1, 2):
            raise ValueError("Estimator only supports 1D binary mediator.")
        raise ValueError("Estimator does not support 1D binary mediator.")
    return effects
