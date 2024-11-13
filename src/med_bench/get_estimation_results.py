#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

from .mediation import (
    mediation_IPW,
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


def get_estimation_results(x, t, m, y, estimator, config):
    """Dynamically selects and calls an estimator (class-based or legacy function) to estimate total, direct, and indirect effects."""

    effects = None  # Initialize variable to store the effects

    # Helper function for regularized regressor and classifier initialization
    def get_regularized_regressor_and_classifier(regularize=True, calibration=None, method="sigmoid"):
        cs, alphas = _get_regularization_parameters(regularization=regularize)
        clf = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)
        reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)
        if calibration:
            clf = CalibratedClassifierCV(clf, method=method)
        return clf, reg

    if estimator == "mediation_IPW_R":
        # Use R-based mediation estimator with direct output extraction
        x_r, t_r, m_r, y_r = [_convert_array_to_R(uu) for uu in (x, t, m, y)]
        output_w = causalweight.medweight(
            y=y_r, d=t_r, m=m_r, x=x_r, trim=0.0, ATET="FALSE", logit="TRUE", boot=2
        )
        raw_res_R = np.array(output_w.rx2("results"))
        effects = raw_res_R[0, :]

    elif estimator == "coefficient_product":
        # Class-based implementation for CoefficientProduct
        estimator_obj = CoefficientProduct(regularize=True)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_noreg":
        # Class-based implementation for InversePropensityWeighting without regularization
        clf, reg = get_regularized_regressor_and_classifier(regularize=False)
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_noreg_cf":
        # Legacy function for crossfit with no regularization
        effects = mediation_IPW(
            y, t, m, x, trim=0, regularization=False, forest=False, crossfit=2, clip=1e-6, calibration=None
        )

    elif estimator == "mediation_ipw_reg":
        # Class-based implementation with regularization
        clf, reg = get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_reg_cf":
        # Legacy function with crossfit and regularization
        effects = mediation_IPW(
            y, t, m, x, trim=0, regularization=True, forest=False, crossfit=2, clip=1e-6, calibration=None
        )

    elif estimator == "mediation_ipw_reg_calibration":
        # Class-based implementation with regularization and calibration (sigmoid)
        clf, reg = get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="sigmoid")
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_reg_calibration_iso":
        # Class-based implementation with isotonic calibration
        clf, reg = get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="isotonic")
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_reg_calibration_cf":
        # Legacy function with crossfit and sigmoid calibration
        effects = mediation_IPW(
            y, t, m, x, trim=0, regularization=True, forest=False, crossfit=2, clip=1e-6, calibration="sigmoid"
        )

    elif estimator == "mediation_ipw_reg_calibration_iso_cf":
        # Legacy function with crossfit and isotonic calibration
        effects = mediation_IPW(
            y, t, m, x, trim=0, regularization=True, forest=False, crossfit=2, clip=1e-6, calibration="isotonic"
        )

    elif estimator == "mediation_ipw_forest":
        # Class-based implementation with forest models
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_forest_cf":
        # Legacy function with forest and crossfit
        effects = mediation_IPW(
            y, t, m, x, trim=0, regularization=True, forest=True, crossfit=2, clip=1e-6, calibration=None
        )

    elif estimator == "mediation_ipw_forest_calibration":
        # Class-based implementation with forest and calibrated sigmoid
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid")
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=calibrated_clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_ipw_forest_calibration_iso":
        # Class-based implementation with isotonic calibration
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        calibrated_clf = CalibratedClassifierCV(clf, method="isotonic")
        estimator_obj = InversePropensityWeighting(
            clip=1e-6, trim=0, regressor=reg, classifier=calibrated_clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_noreg":
        # Class-based implementation of GComputation without regularization
        clf, reg = get_regularized_regressor_and_classifier(regularize=False)
        estimator_obj = GComputation(regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_noreg_cf":
        # Legacy function with crossfit and no regularization
        effects = mediation_g_formula(
            y, t, m, x, interaction=False, forest=False, crossfit=2, regularization=False, calibration=None
        )

    elif estimator == "mediation_g_computation_reg":
        # Class-based implementation of GComputation with regularization
        clf, reg = get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = GComputation(regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_reg_cf":
        # Legacy function with regularization and crossfit
        effects = mediation_g_formula(
            y, t, m, x, interaction=False, forest=False, crossfit=2, regularization=True, calibration=None
        )

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

    elif estimator == "mediation_g_computation_reg_calibration":
        # Class-based implementation with regularization and calibrated sigmoid
        clf, reg = get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="sigmoid")
        estimator_obj = GComputation(regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_reg_calibration_iso":
        # Class-based implementation with isotonic calibration
        clf, reg = get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="isotonic")
        estimator_obj = GComputation(regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_g_computation_forest":
        # Class-based implementation with forest models
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator_obj = GComputation(regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "mediation_multiply_robust_noreg":
        # Class-based implementation for MultiplyRobust without regularization
        clf, reg = get_regularized_regressor_and_classifier(regularize=False)
        estimator_obj = MultiplyRobust(
            ratio="propensities", normalized=True, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    elif estimator == "simulation_based":
        # R-based function for simulation
        effects = r_mediate(y, t, m, x, interaction=False)

    elif estimator == "mediation_dml":
        # R-based function for Double Machine Learning with legacy config
        effects = r_mediation_dml(y, t, m, x, trim=0.0, order=1)

    elif estimator == "mediation_dml_noreg":
        # Class-based implementation for DoubleMachineLearning without regularization
        clf, reg = get_regularized_regressor_and_classifier(regularize=False)
        estimator_obj = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # Regularized, crossfitting, calibration (isotonic) for InversePropensityWeighting
    elif estimator == "mediation_ipw_reg_calibration_iso_cf":
        effects = mediation_IPW(
            y, t, m, x, trim=0, regularization=True, forest=False, crossfit=2, clip=1e-6, calibration="isotonic"
        )

    # Forest and crossfit with sigmoid calibration for InversePropensityWeighting
    elif estimator == "mediation_ipw_forest_calibration_cf":
        effects = mediation_IPW(
            y, t, m, x, trim=0, regularization=True, forest=True, crossfit=2, clip=1e-6, calibration="sigmoid"
        )

    # Forest and crossfit with isotonic calibration for InversePropensityWeighting
    elif estimator == "mediation_ipw_forest_calibration_iso_cf":
        effects = mediation_IPW(
            y, t, m, x, trim=0, regularization=True, forest=True, crossfit=2, clip=1e-6, calibration="isotonic"
        )

    # MultiplyRobust without regularization, with crossfitting
    elif estimator == "mediation_multiply_robust_noreg_cf":
        effects = mediation_multiply_robust(
            y, t, m.astype(int), x, interaction=False, forest=False, crossfit=2, clip=1e-6, regularization=False, calibration=None
        )

    # Regularized MultiplyRobust estimator
    elif estimator == "mediation_multiply_robust_reg":
        clf, reg = get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = MultiplyRobust(
            ratio="propensities", normalized=True, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # Regularized MultiplyRobust with crossfitting
    elif estimator == "mediation_multiply_robust_reg_cf":
        effects = mediation_multiply_robust(
            y, t, m.astype(int), x, interaction=False, forest=False, crossfit=2, clip=1e-6, regularization=True, calibration=None
        )

    # Regularized MultiplyRobust with sigmoid calibration
    elif estimator == "mediation_multiply_robust_reg_calibration":
        clf, reg = get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="sigmoid")
        estimator_obj = MultiplyRobust(
            ratio="propensities", normalized=True, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # Regularized MultiplyRobust with isotonic calibration
    elif estimator == "mediation_multiply_robust_reg_calibration_iso":
        clf, reg = get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="isotonic")
        estimator_obj = MultiplyRobust(
            ratio="propensities", normalized=True, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # Regularized MultiplyRobust with sigmoid calibration and crossfitting
    elif estimator == "mediation_multiply_robust_reg_calibration_cf":
        effects = mediation_multiply_robust(
            y, t, m.astype(int), x, interaction=False, forest=False, crossfit=2, clip=1e-6, regularization=True, calibration="sigmoid"
        )

    # Regularized MultiplyRobust with isotonic calibration and crossfitting
    elif estimator == "mediation_multiply_robust_reg_calibration_iso_cf":
        effects = mediation_multiply_robust(
            y, t, m.astype(int), x, interaction=False, forest=False, crossfit=2, clip=1e-6, regularization=True, calibration="isotonic"
        )

    elif estimator == "mediation_multiply_robust_forest":
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator = MultiplyRobust(
            ratio="propensities", normalized=True, regressor=reg,
            classifier=clf)
        estimator.fit(t, m, x, y)
        causal_effects = estimator.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # MultiplyRobust with forest and crossfitting
    elif estimator == "mediation_multiply_robust_forest_cf":
        effects = mediation_multiply_robust(
            y, t, m.astype(int), x, interaction=False, forest=True, crossfit=2, clip=1e-6, regularization=True, calibration=None
        )

    # MultiplyRobust with forest and sigmoid calibration
    elif estimator == "mediation_multiply_robust_forest_calibration":
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid")
        estimator_obj = MultiplyRobust(
            ratio="propensities", normalized=True, regressor=reg, classifier=calibrated_clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # MultiplyRobust with forest and isotonic calibration
    elif estimator == "mediation_multiply_robust_forest_calibration_iso":
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        calibrated_clf = CalibratedClassifierCV(clf, method="isotonic")
        estimator_obj = MultiplyRobust(
            ratio="propensities", normalized=True, regressor=reg, classifier=calibrated_clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # MultiplyRobust with forest, sigmoid calibration, and crossfitting
    elif estimator == "mediation_multiply_robust_forest_calibration_cf":
        effects = mediation_multiply_robust(
            y, t, m.astype(int), x, interaction=False, forest=True, crossfit=2, clip=1e-6, regularization=True, calibration="sigmoid"
        )

    # MultiplyRobust with forest, isotonic calibration, and crossfitting
    elif estimator == "mediation_multiply_robust_forest_calibration_iso_cf":
        effects = mediation_multiply_robust(
            y, t, m.astype(int), x, interaction=False, forest=True, crossfit=2, clip=1e-6, regularization=True, calibration="isotonic"
        )

    # Regularized Double Machine Learning
    elif estimator == "mediation_dml_reg":
        clf, reg = get_regularized_regressor_and_classifier(regularize=True)
        estimator_obj = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # Double Machine Learning with fixed seed
    elif estimator == "mediation_dml_reg_fixed_seed":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, random_state=321, calibration=None)

    # Double Machine Learning without regularization, with crossfitting
    elif estimator == "mediation_dml_noreg_cf":
        effects = mediation_dml(y, t, m, x, trim=0, clip=1e-6,
                                crossfit=2, regularization=False, calibration=None)

    # Regularized Double Machine Learning with crossfitting
    elif estimator == "mediation_dml_reg_cf":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, crossfit=2, calibration=None)

    # Regularized Double Machine Learning with sigmoid calibration
    elif estimator == "mediation_dml_reg_calibration":
        clf, reg = get_regularized_regressor_and_classifier(
            regularize=True, calibration=True, method="sigmoid")
        estimator_obj = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # Regularized Double Machine Learning with forest models
    elif estimator == "mediation_dml_forest":
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        estimator_obj = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # Double Machine Learning with forest and calibrated sigmoid
    elif estimator == "mediation_dml_forest_calibration":
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid")
        estimator_obj = DoubleMachineLearning(
            clip=1e-6, trim=0, normalized=True, regressor=reg, classifier=calibrated_clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # Double Machine Learning with forest, crossfitting, and sigmoid calibration
    elif estimator == "mediation_dml_reg_calibration_cf":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, crossfit=2, calibration="sigmoid", forest=False)

    # Double Machine Learning with forest and crossfitting
    elif estimator == "mediation_dml_forest_cf":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, crossfit=2, calibration=None, forest=True)

    # Double Machine Learning with forest, crossfitting, and calibrated sigmoid
    elif estimator == "mediation_dml_forest_calibration_cf":
        effects = mediation_dml(
            y, t, m, x, trim=0, clip=1e-6, crossfit=2, calibration="sigmoid", forest=True)

    # GComputation with regularization, crossfitting, and sigmoid calibration
    elif estimator == "mediation_g_computation_reg_calibration_cf":
        effects = mediation_g_formula(
            y, t, m, x, interaction=False, forest=False, crossfit=2, regularization=True, calibration="sigmoid")

    # GComputation with forest and sigmoid calibration
    elif estimator == "mediation_g_computation_forest_calibration":
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid")
        estimator_obj = GComputation(regressor=reg, classifier=calibrated_clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # GComputation with forest and isotonic calibration
    elif estimator == "mediation_g_computation_forest_calibration_iso":
        clf = RandomForestClassifier(
            random_state=42, n_estimators=100, min_samples_leaf=10)
        reg = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42)
        calibrated_clf = CalibratedClassifierCV(clf, method="isotonic")
        estimator_obj = GComputation(regressor=reg, classifier=calibrated_clf)
        estimator_obj.fit(t, m, x, y)
        causal_effects = estimator_obj.estimate(t, m, x, y)
        effects = transform_outputs(causal_effects)

    # GComputation with forest, crossfitting, and sigmoid calibration
    elif estimator == "mediation_g_computation_forest_calibration_cf":
        effects = mediation_g_formula(
            y, t, m, x, interaction=False, forest=True, crossfit=2, regularization=True, calibration="sigmoid")

    # GComputation with forest, crossfitting, and isotonic calibration
    elif estimator == "mediation_g_computation_forest_calibration_iso_cf":
        effects = mediation_g_formula(
            y, t, m, x, interaction=False, forest=True, crossfit=2, regularization=True, calibration="isotonic")

    elif estimator == "mediation_g_estimator":
        if config in (0, 1, 2):
            effects = r_mediation_g_estimator(y, t, m, x)
    else:
        raise ValueError("Unrecognized estimator label.")

    # Catch unsupported estimators and raise an error
    if effects is None:
        raise ValueError(
            f"Estimation failed for {estimator}. Check inputs or configuration.")
    return effects
