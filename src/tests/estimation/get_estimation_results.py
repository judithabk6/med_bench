#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

from med_bench.estimation.mediation_coefficient_product import CoefficientProduct
from med_bench.estimation.mediation_g_computation import GComputation
from med_bench.estimation.mediation_ipw import InversePropensityWeighting
from med_bench.estimation.mediation_mr import MultiplyRobust
from med_bench.estimation.mediation_tmle import TMLE
from med_bench.utils.utils import _get_regularization_parameters
from med_bench.utils.constants import CV_FOLDS

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.calibration import CalibratedClassifierCV


def _transform_outputs(causal_effects):
    """Transforms outputs in the old format

    Args:
        causal_effects (dict): dictionary of causal effects

    Returns:
        list: list of causal effects
    """
    total = causal_effects["total_effect"]
    direct_treated = causal_effects["direct_effect_treated"]
    direct_control = causal_effects["direct_effect_control"]
    indirect_treated = causal_effects["indirect_effect_treated"]
    indirect_control = causal_effects["indirect_effect_control"]
    return np.array(
        [total, direct_treated, direct_control, indirect_treated, indirect_control]
    ).astype(float)


def _get_estimation_results(x, t, m, y, estimator):
    """Dynamically selects and calls an estimator (class-based or legacy function) to estimate total, direct, and indirect effects."""

    effects = None  # Initialize variable to store the effects

    # Helper function for regularized regressor and classifier initialization
    def _get_regularized_regressor_and_classifier(
        regularize=True, calibration=None, method="sigmoid"
    ):
        cs, alphas = _get_regularization_parameters(regularization=regularize)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        if calibration:
            clf = CalibratedClassifierCV(clf, method=method)
        return clf, reg

    if estimator == "coefficient_product":
        # Class-based implementation for CoefficientProduct
        estimator_obj = CoefficientProduct(regularize=True)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_reg_prop_ratio_treatment":
        # Class-based implementation with regularization
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, classifier=clf, prop_ratio="treatment"
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_reg_calibration_prop_ratio_treatment":
        # Class-based implementation with regularization and calibration (sigmoid)
        clf, reg = _get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="sigmoid"
        )
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, classifier=clf, prop_ratio="treatment"
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_reg_prop_ratio_mediator":
        # Class-based implementation with regularization
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, classifier=clf, prop_ratio="mediator"
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_reg_prop_ratio_treatment_cross_fit":
        # Class-based implementation with regularization
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, classifier=clf, prop_ratio="treatment"
        )
        causal_effects = estimator_obj.cross_fit_estimate(t, m, x, y, n_splits=2)
        effects = _transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_reg_integ_implicit":
        # Class-based implementation of GComputation with regularization
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = GComputation(
            regressor=reg, classifier=clf, integration="implicit"
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_reg_calibration_integ_implicit":
        # Class-based implementation with regularization and calibrated sigmoid
        clf, reg = _get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="sigmoid"
        )
        estimator_obj = GComputation(
            regressor=reg, classifier=clf, integration="implicit"
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_reg_integ_explicit":
        # Class-based implementation of GComputation with regularization
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = GComputation(
            regressor=reg, classifier=clf, integration="explicit"
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_reg_integ_implicit_cross_fit":
        # Class-based implementation of GComputation with regularization
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = GComputation(
            regressor=reg, classifier=clf, integration="implicit"
        )
        causal_effects = estimator_obj.cross_fit_estimate(t, m, x, y, n_splits=2)
        effects = _transform_outputs(causal_effects)

    # Regularized MultiplyRobust estimator
    elif estimator == "mediation_multiply_robust_reg_pr_treatment_ig_implicit":
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = MultiplyRobust(
            clip=1e-6,
            trim=0,
            prop_ratio="treatment",
            normalized=True,
            regressor=reg,
            classifier=clf,
            integration="implicit",
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    # Regularized MultiplyRobust with sigmoid calibration
    elif (
        estimator
        == "mediation_multiply_robust_reg_calibration_pr_treatment_ig_implicit"
    ):
        clf, reg = _get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="sigmoid"
        )
        estimator_obj = MultiplyRobust(
            clip=1e-6,
            trim=0,
            prop_ratio="treatment",
            normalized=True,
            regressor=reg,
            classifier=clf,
            integration="implicit",
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    # Regularized MultiplyRobust estimator
    elif estimator == "mediation_multiply_robust_reg_pr_mediator_ig_implicit":
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = MultiplyRobust(
            clip=1e-6,
            trim=0,
            prop_ratio="mediator",
            normalized=True,
            regressor=reg,
            classifier=clf,
            integration="implicit",
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    # Regularized MultiplyRobust estimator
    elif estimator == "mediation_multiply_robust_reg_pr_treatment_ig_explicit":
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = MultiplyRobust(
            clip=1e-6,
            trim=0,
            prop_ratio="treatment",
            normalized=True,
            regressor=reg,
            classifier=clf,
            integration="explicit",
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    # Regularized MultiplyRobust estimator
    elif estimator == "mediation_multiply_robust_reg_pr_mediator_ig_explicit":
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = MultiplyRobust(
            clip=1e-6,
            trim=0,
            prop_ratio="mediator",
            normalized=True,
            regressor=reg,
            classifier=clf,
            integration="explicit",
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    # Regularized MultiplyRobust estimator with cross fitting
    elif (
        estimator == "mediation_multiply_robust_reg_pr_treatment_ig_implicit_cross_fit"
    ):
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = MultiplyRobust(
            clip=1e-6,
            trim=0,
            prop_ratio="treatment",
            normalized=True,
            regressor=reg,
            classifier=clf,
            integration="implicit",
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.cross_fit_estimate(t, m, x, y, n_splits=2)
        effects = _transform_outputs(causal_effects)

    # TMLE - ratio of treatment propensities
    elif estimator == "mediation_tmle_prop_ratio_treatment":
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = TMLE(
            clip=1e-6, trim=0, regressor=reg, classifier=clf, prop_ratio="treatment"
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    # TMLE - ratio of mediator densities
    elif estimator == "mediation_tmle_prop_ratio_mediator":
        clf, reg = _get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = TMLE(
            clip=1e-6, trim=0, regressor=reg, classifier=clf, prop_ratio="mediator"
        )
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = _transform_outputs(causal_effects)

    else:
        raise ValueError("Unrecognized estimator label for {}.".format(estimator))

    # Catch unsupported estimators and raise an error
    if effects is None:
        raise ValueError(
            f"Estimation failed for {estimator}. Check inputs or configuration."
        )
    return effects
