"""
get_estimation.py::get_estimation

We test :
-

We pinpoint _


"""

from pprint import pprint
import pytest
import numpy as np
from numpy.random import default_rng
from med_bench.src.get_simulated_data import simulate_data
from med_bench.src.get_estimation import get_estimation


@pytest.fixture(autouse=True)
def data():
    n = 3
    rg = default_rng(42)
    mis_spec_m = False
    mis_spec_y = False
    dim_x = 1
    dim_m = 1
    seed = 1
    type_m = "binary"
    sigma_y = 0.5
    sigma_m = 0.5
    beta_t_factor = 10
    beta_m_factor = 1

    return simulate_data(
        n,
        rg,
        mis_spec_m,
        mis_spec_y,
        dim_x,
        dim_m,
        seed,
        type_m,
        sigma_y,
        sigma_m,
        beta_t_factor,
        beta_m_factor,
    )


@pytest.fixture(autouse=True)
def effects():
    return np.array(data[4:9])


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
