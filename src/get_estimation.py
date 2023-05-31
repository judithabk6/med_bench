#!/usr/bin/env python
# -*- coding:utf-8 -*-


import time
from .benchmark_mediation import *
from rpy2.rinterface import RRuntimeError
import sys
import pandas as pd
import numpy as np


def get_estimation(x, t, m, y, estimator, config):
    if estimator == 'huber_IPW_R':
        x_r, t_r, m_r, y_r = [_convert_array_to_R(uu) for uu in (x, t, m, y)]
        output_w = causalweight.medweight(y=y_r, d=t_r, m=m_r, x=x_r, trim=0.0,
                                          ATET="FALSE", logit="TRUE", boot=2)
        raw_res_R = np.array(output_w.rx2('results'))
        effects = raw_res_R[0, :]
    if estimator == 'coefficient_product':
        effects = ols_mediation(y, t, m, x)
    if estimator == 'huber_ipw_noreg':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=False, forest=False, crossfit=0,
                            clip=0.0, calibration=False)
    if estimator == 'huber_ipw_noreg_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=False, forest=False, crossfit=2,
                            clip=0.0, calibration=False)
    if estimator == 'huber_ipw_reg':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=0,
                            clip=0.0, calibration=False)
    if estimator == 'huber_ipw_reg_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=2,
                            clip=0.0, calibration=False)
    if estimator == 'huber_ipw_reg_calibration':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=0,
                            clip=0.0, calibration=True)
    if estimator == 'huber_ipw_reg_calibration_iso':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=0,
                            clip=0.0, calibration=True,
                            calib_method='isotonic')
    if estimator == 'huber_ipw_reg_calibration_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=2,
                            clip=0.0, calibration=True)
    if estimator == 'huber_ipw_reg_calibration_iso_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=2,
                            clip=0.0, calibration=True,
                            calib_method='isotonic')
    if estimator == 'huber_ipw_forest':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=0,
                            clip=0.0, calibration=False)
    if estimator == 'huber_ipw_forest_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=2,
                            clip=0.0, calibration=False)
    if estimator == 'huber_ipw_forest_calibration':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=0,
                            clip=0.0, calibration=True)
    if estimator == 'huber_ipw_forest_calibration_iso':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=0,
                            clip=0.0, calibration=True,
                            calib_method='isotonic')
    if estimator == 'huber_ipw_forest_calibration_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=2,
                            clip=0.0, calibration=True)
    if estimator == 'huber_ipw_forest_calibration_iso_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=2,
                            clip=0.0, calibration=True,
                            calib_method='isotonic')
    if estimator == 'g_computation_noreg':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=0,
                                    regularization=False, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_noreg_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=2,
                                    regularization=False, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=0,
                                    regularization=True, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=2,
                                    regularization=True, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_calibration':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=0,
                                    regularization=True, calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_calibration_iso':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=0,
                                    regularization=True, calibration=True,
                                    calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_calibration_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=2,
                                    regularization=True, calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_calibration_iso_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=2,
                                    regularization=True, calibration=True,
                                    calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=0,
                                    regularization=True, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=2,
                                    regularization=True, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_calibration':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=0,
                                    regularization=True, calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_calibration_iso':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=0,
                                    regularization=True, calibration=True,
                                    calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_calibration_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=2,
                                    regularization=True, calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_calibration_iso_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=2,
                                    regularization=True, calibration=True,
                                    calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_noreg':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=0, clip=0.0,
                                                regularization=False,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_noreg_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=2, clip=0.0,
                                                regularization=False,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_calibration':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_calibration_iso':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=True,
                                                calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_calibration_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_calibration_iso_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=True,
                                                calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_calibration':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_calibration_iso':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=True,
                                                calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_calibration_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_calibration_iso_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=True,
                                                calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'simulation_based':
        if config in (0, 1, 2):
            effects = r_mediate(y, t, m, x, interaction=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'DML_huber':
        if config > 0:
            effects = medDML(y, t, m, x, trim=0.0, order=1)
        else:
            effects = [np.nan] * 5
    if estimator == 'G_estimator':
        if config in (0, 1, 2):
            effects = g_estimator(y, t, m, x)
        else:
            effects = [np.nan] * 5
    return effects

