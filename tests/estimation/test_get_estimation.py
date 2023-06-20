"""
get_estimation.py::get_estimation

We test all the benchmark_mediation estimators for a certain tolerance.
The test is skipped if estimator has not been implemented yet,
i.e. all outputs are NaN or NotImplementedError is raised.
The test xfails for any other wierd behavior.

We pinpoint that :
- DML_huber is not working, RRuntimeError is raised
- multiply_robust methods return some NaN
- A few methods get a relative error of more than 100%,
even in the linear framework
- t.ravel() and y.ravel() are necessary to get IPW proper behavior
"""

from pprint import pprint
import pytest
import numpy as np
from numpy.random import default_rng
from med_bench.src.get_simulated_data import simulate_data
from med_bench.src.get_estimation import get_estimation
import seaborn as sns
import matplotlib.pyplot as plt


ATE_TOLERANCE = 0.2
DIRECT_TOELERANCE = 0.2
INDIRECT_TOLERANCE = 0.6  # indirect effect is weak leading to a huge relative error
TOLERANCE = np.array(
    [
        ATE_TOLERANCE,
        DIRECT_TOELERANCE,
        DIRECT_TOELERANCE,
        INDIRECT_TOLERANCE,
        INDIRECT_TOLERANCE,
    ]
)


data = simulate_data(
    n=1000,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=5,
    dim_m=1,
    seed=1,
    type_m="binary",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=0.5,
    beta_m_factor=0.5,
)

x = data[0]
t = data[1]
m = data[2]
y = data[3]
effects = np.array(data[4:9])

estimator_list = [
    "coefficient_product",
    "huber_ipw_noreg",
    "huber_ipw_reg",
    "huber_ipw_reg_calibration",
    "huber_ipw_forest",
    "huber_ipw_forest_calibration",
    "g_computation_noreg",
    "g_computation_reg",
    "g_computation_reg_calibration",
    "g_computation_forest",
    "g_computation_forest_calibration",
    "multiply_robust_noreg",
    "multiply_robust_reg",
    "multiply_robust_reg_calibration",
    "multiply_robust_forest",
    "multiply_robust_forest_calibration",
    "simulation_based",
    "DML_huber",
    "G_estimator",
    "huber_ipw_noreg_cf",
    "huber_ipw_reg_cf",
    "huber_ipw_reg_calibration_cf",
    "huber_ipw_forest_cf",
    "huber_ipw_forest_calibration_cf",
    "g_computation_noreg_cf",
    "g_computation_reg_cf",
    "g_computation_reg_calibration_cf",
    "g_computation_forest_cf",
    "g_computation_forest_calibration_cf",
    "multiply_robust_noreg_cf",
    "multiply_robust_reg_cf",
    "multiply_robust_reg_calibration_cf",
    "multiply_robust_forest_cf",
    "multiply_robust_forest_calibration_cf",
]


@pytest.mark.parametrize("estimator", estimator_list)
@pytest.mark.parametrize("config", [1, 5])
@pytest.mark.parametrize("error_tolerance", [TOLERANCE])
def test_tolerance(estimator, config, error_tolerance):
    try:
        effects_chap = get_estimation(x, t.ravel(), m, y.ravel(), estimator, config)
        # effects_chap = get_estimation(x, t, m, y, estimator, config)
        effects_chap = effects_chap[0:5]
    except Exception as get_estimation_error:
        if get_estimation_error != NotImplementedError:
            pytest.xfail("Missing NotImplementedError")
        else:
            pytest.mark.skip("Not implemented")
    else:
        error = abs((effects_chap - effects) / effects)
        if np.all(np.isnan(effects_chap)):
            pytest.skip("all effects are NaN")
        elif np.any(np.isnan(effects_chap)):
            pprint("NaN found")
            assert np.all(error[~np.isnan(error)] <= error_tolerance[~np.isnan(error)])
        else:
            assert np.all(error <= error_tolerance)
            # pprint(
            #     f"Relative error estimate = {error}. Error tolerance = {error_tolerance}%."
            # )
            # pprint(f"Estimated = {effects_chap}. True = {effects}.")
