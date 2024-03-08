#!/usr/bin/env python
# -*- coding:utf-8 -*-


import time
import sys
# from rpy2.rinterface_lib.embedded import RRuntimeError
import pandas as pd
import numpy as np
from .mediation import (
    mediation_IPW,
    mediation_coefficient_product,
    mediation_g_formula,
    mediation_multiply_robust,
    mediation_DML,
    r_mediation_g_estimator,
    r_mediation_DML,
    r_mediate,
)

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
            clip=0.0,
            calibration=None,
        )
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
            clip=0.0,
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
            clip=0.0,
            calibration=None,
        )
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
            clip=0.0,
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
            clip=0.0,
            calibration=None,
        )
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
            clip=0.0,
            calibration="isotonic",
        )
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
            clip=0.0,
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
            clip=0.0,
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
            clip=0.0,
            calibration=None,
        )
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
            clip=0.0,
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
            clip=0.0,
            calibration=None,
        )
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
            clip=0.0,
            calibration="isotonic",
        )
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
            clip=0.0,
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
            clip=0.0,
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
                clip=0.0,
                regularization=False,
                calibration=None,
            )
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
                clip=0.0,
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
                clip=0.0,
                regularization=True,
                calibration=None,
            )
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
                clip=0.0,
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
                clip=0.0,
                regularization=True,
                calibration='sigmoid',
            )
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
                clip=0.0,
                regularization=True,
                calibration="isotonic",
            )
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
                clip=0.0,
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
                clip=0.0,
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
                clip=0.0,
                regularization=True,
                calibration=None,
            )
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
                clip=0.0,
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
                clip=0.0,
                regularization=True,
                calibration='sigmoid',
            )
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
                clip=0.0,
                regularization=True,
                calibration="isotonic",
            )
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
                clip=0.0,
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
                clip=0.0,
                regularization=True,
                calibration="isotonic",
            )
    elif estimator == "simulation_based":
        if config in (0, 1, 2):
            effects = r_mediate(y, t, m, x, interaction=False)
    elif estimator == "mediation_DML":
        if config > 0:
            effects = r_mediation_DML(y, t, m, x, trim=0.0, order=1)
    elif estimator == "mediation_DML_noreg":
        effects = mediation_DML(
            y, t, m, x, trim=0, regularization=False, calibration=None)
    elif estimator == "mediation_DML_reg":
        effects = mediation_DML(y, t, m, x, trim=0, calibration=None)
    elif estimator == "mediation_DML_reg_fixed_seed":
        effects = mediation_DML(
            y, t, m, x, trim=0, random_state=321, calibration=None)
    elif estimator == "mediation_DML_noreg_cf":
        effects = mediation_DML(
            y,
            t,
            m,
            x,
            trim=0,
            crossfit=2,
            regularization=False,
            calibration=None)
    elif estimator == "mediation_DML_reg_cf":
        effects = mediation_DML(
            y, t, m, x, trim=0, crossfit=2, calibration=None)
    elif estimator == "mediation_DML_reg_calibration":
        effects = mediation_DML(
            y, t, m, x, trim=0, crossfit=0, calibration='sigmoid')
    elif estimator == "mediation_DML_forest":
        effects = mediation_DML(
            y, t, m, x, trim=0, crossfit=0, calibration=None, forest=True)
    elif estimator == "mediation_DML_forest_calibration":
        effects = mediation_DML(
            y, t, m, x, trim=0, crossfit=0, calibration='sigmoid', forest=True)
    elif estimator == "mediation_DML_reg_calibration_cf":
        effects = mediation_DML(
            y, t, m, x, trim=0, crossfit=2, calibration='sigmoid', forest=False)
    elif estimator == "mediation_DML_forest_cf":
        effects = mediation_DML(
            y, t, m, x, trim=0, crossfit=2, calibration=None, forest=True)
    elif estimator == "mediation_DML_forest_calibration_cf":
        effects = mediation_DML(
            y, t, m, x, trim=0, crossfit=2, calibration='sigmoid', forest=True)
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
