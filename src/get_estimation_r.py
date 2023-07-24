#!/usr/bin/env python
# -*- coding:utf-8 -*-


import time
import sys
from rpy2.rinterface_lib.embedded import RRuntimeError
import pandas as pd
import numpy as np
from .estimators_r import *


def get_estimation_r(x, t, m, y, estimator, config):
    """Wrapper estimator fonction ; calls an estimator given mediation data
    in order to estimate total, direct, and indirect effects, for estimators
    implemented in R.

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
    if estimator == "huber_IPW_R":
        x_r, t_r, m_r, y_r = [_convert_array_to_R(uu) for uu in (x, t, m, y)]
        output_w = causalweight.medweight(
            y=y_r, d=t_r, m=m_r, x=x_r, trim=0.0, ATET="FALSE", logit="TRUE", boot=2
        )
        raw_res_R = np.array(output_w.rx2("results"))
        effects = raw_res_R[0, :]
    elif estimator == "simulation_based":
        if config in (0, 1, 2):
            effects = r_mediate(y, t, m, x, interaction=False)
    elif estimator == "DML_huber":
        if config > 0:
            effects = medDML(y, t, m, x, trim=0.0, order=1)
    elif estimator == "G_estimator":
        if config in (0, 1, 2):
            effects = g_estimator(y, t, m, x)
    else:
        raise ValueError("Unrecognized estimator label.")
    if effects is None:
        if config not in (0, 1, 2):
            raise ValueError("Estimator only supports 1D binary mediator.")
        raise ValueError("Estimator does not support 1D binary mediator.")
    return effects
