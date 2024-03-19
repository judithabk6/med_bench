import itertools
import os
import numpy as np
from numpy.random import default_rng

# CONSTANTS USED FOR TESTS

# TOLERANCE THRESHOLDS

SMALL_ATE_TOLERANCE = 0.05
SMALL_DIRECT_TOLERANCE = 0.05
SMALL_INDIRECT_TOLERANCE = 0.2

MEDIUM_ATE_TOLERANCE = 0.10
MEDIUM_DIRECT_TOLERANCE = 0.10
MEDIUM_INDIRECT_TOLERANCE = 0.4

LARGE_ATE_TOLERANCE = 0.15
LARGE_DIRECT_TOLERANCE = 0.15
LARGE_INDIRECT_TOLERANCE = 0.8
# indirect effect is weak, leading to a large relative error

SMALL_TOLERANCE = np.array(
    [
        SMALL_ATE_TOLERANCE,
        SMALL_DIRECT_TOLERANCE,
        SMALL_DIRECT_TOLERANCE,
        SMALL_INDIRECT_TOLERANCE,
        SMALL_INDIRECT_TOLERANCE,
    ]
)

MEDIUM_TOLERANCE = np.array(
    [
        MEDIUM_ATE_TOLERANCE,
        MEDIUM_DIRECT_TOLERANCE,
        MEDIUM_DIRECT_TOLERANCE,
        MEDIUM_INDIRECT_TOLERANCE,
        MEDIUM_INDIRECT_TOLERANCE,
    ]
)

LARGE_TOLERANCE = np.array(
    [
        LARGE_ATE_TOLERANCE,
        LARGE_DIRECT_TOLERANCE,
        LARGE_DIRECT_TOLERANCE,
        LARGE_INDIRECT_TOLERANCE,
        LARGE_INDIRECT_TOLERANCE,
    ]
)

INFINITE_TOLERANCE = np.array(
    [
        np.inf,
        np.inf,
        np.inf,
        np.inf,
        np.inf,
    ]
)

TOLERANCE_DICT = {
    "coefficient_product": LARGE_TOLERANCE,
    "mediation_ipw_noreg": INFINITE_TOLERANCE,
    "mediation_ipw_reg": INFINITE_TOLERANCE,
    "mediation_ipw_reg_calibration": INFINITE_TOLERANCE,
    "mediation_ipw_forest": INFINITE_TOLERANCE,
    "mediation_ipw_forest_calibration": INFINITE_TOLERANCE,
    "mediation_g_computation_noreg": LARGE_TOLERANCE,
    "mediation_g_computation_reg": MEDIUM_TOLERANCE,
    "mediation_g_computation_reg_calibration": LARGE_TOLERANCE,
    "mediation_g_computation_forest": LARGE_TOLERANCE,
    "mediation_g_computation_forest_calibration": INFINITE_TOLERANCE,
    "mediation_multiply_robust_noreg": INFINITE_TOLERANCE,
    "mediation_multiply_robust_reg": LARGE_TOLERANCE,
    "mediation_multiply_robust_reg_calibration": LARGE_TOLERANCE,
    "mediation_multiply_robust_forest": INFINITE_TOLERANCE,
    "mediation_multiply_robust_forest_calibration": LARGE_TOLERANCE,
    "simulation_based": LARGE_TOLERANCE,
    "mediation_DML": INFINITE_TOLERANCE,
    "mediation_DML_reg_fixed_seed": INFINITE_TOLERANCE,
    "mediation_g_estimator": SMALL_TOLERANCE,
    "mediation_ipw_noreg_cf": INFINITE_TOLERANCE,
    "mediation_ipw_reg_cf": INFINITE_TOLERANCE,
    "mediation_ipw_reg_calibration_cf": INFINITE_TOLERANCE,
    "mediation_ipw_forest_cf": INFINITE_TOLERANCE,
    "mediation_ipw_forest_calibration_cf": INFINITE_TOLERANCE,
    "mediation_g_computation_noreg_cf": SMALL_TOLERANCE,
    "mediation_g_computation_reg_cf": LARGE_TOLERANCE,
    "mediation_g_computation_reg_calibration_cf": LARGE_TOLERANCE,
    "mediation_g_computation_forest_cf": INFINITE_TOLERANCE,
    "mediation_g_computation_forest_calibration_cf": LARGE_TOLERANCE,
    "mediation_multiply_robust_noreg_cf": MEDIUM_TOLERANCE,
    "mediation_multiply_robust_reg_cf": LARGE_TOLERANCE,
    "mediation_multiply_robust_reg_calibration_cf": MEDIUM_TOLERANCE,
    "mediation_multiply_robust_forest_cf": INFINITE_TOLERANCE,
    "mediation_multiply_robust_forest_calibration_cf": INFINITE_TOLERANCE,
}

ESTIMATORS = [
    "coefficient_product",
    "mediation_ipw_noreg",
    "mediation_ipw_reg",
    "mediation_ipw_reg_calibration",
    "mediation_ipw_forest",
    "mediation_ipw_forest_calibration",
    "mediation_g_computation_noreg",
    "mediation_g_computation_reg",
    "mediation_g_computation_reg_calibration",
    "mediation_g_computation_forest",
    "mediation_g_computation_forest_calibration",
    "mediation_multiply_robust_noreg",
    "mediation_multiply_robust_reg",
    "mediation_multiply_robust_reg_calibration",
    "mediation_multiply_robust_forest",
    "mediation_multiply_robust_forest_calibration",
    "simulation_based",
    "mediation_DML",
    "mediation_DML_reg_fixed_seed",
    "mediation_g_estimator",
    "mediation_ipw_noreg_cf",
    "mediation_ipw_reg_cf",
    "mediation_ipw_reg_calibration_cf",
    "mediation_ipw_forest_cf",
    "mediation_ipw_forest_calibration_cf",
    "mediation_g_computation_noreg_cf",
    "mediation_g_computation_reg_cf",
    "mediation_g_computation_reg_calibration_cf",
    "mediation_g_computation_forest_cf",
    "mediation_g_computation_forest_calibration_cf",
    "mediation_multiply_robust_noreg_cf",
    "mediation_multiply_robust_reg_cf",
    "mediation_multiply_robust_reg_calibration_cf",
    "mediation_multiply_robust_forest_cf",
    "mediation_multiply_robust_forest_calibration_cf",
]

# PARAMETERS VALUES FOR DATA GENERATION

PARAMETER_NAME = [
    "n",
    "rg",
    "mis_spec_m",
    "mis_spec_y",
    "dim_x",
    "dim_m",
    "seed",
    "type_m",
    "sigma_y",
    "sigma_m",
    "beta_t_factor",
    "beta_m_factor",
]

PARAMETER_LIST = list(
    itertools.product(
        [1000],
        [default_rng(321)],
        [False],
        [False],
        [1, 5],
        [1],
        [123],
        ["binary"],
        [0.5],
        [0.5],
        [0.5],
        [0.5],
    )
)

PARAMETER_LIST.extend(
    list(
        itertools.product(
            [1000],
            [default_rng(321)],
            [False],
            [False],
            [1, 5],
            [1, 5],
            [123],
            ["continuous"],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
        )
    )
)
