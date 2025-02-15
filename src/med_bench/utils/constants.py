import itertools
import numpy as np
from numpy.random import default_rng

# CONSTANTS USED FOR TESTS

# TOLERANCE THRESHOLDS
DEFAULT_TOLERANCE = np.array([0.05, 0.05, 0.05, 0.2, 0.2])

TOLERANCE_FACTOR_DICT = {
    "coefficient_product-M1D_binary_1DX": np.array([1, 1, 1, 4, 4]),
    "coefficient_product-M1D_binary_5DX": np.array([1, 1, 1, 3.5, 3.5]),
    "coefficient_product-M5D_continuous_1DX": np.array([1, 1, 1, 1.5, 1.5]),
    "coefficient_product-M5D_continuous_5DX": np.array([1, 1, 1, 3.5, 3.5]),
    "mediation_ipw_reg_prop_ratio_treatment-M1D_binary_1DX": np.array([6, 1, 1, 100, 100]),
    "mediation_ipw_reg_prop_ratio_treatment-M1D_binary_5DX": np.array([2, 1.2, 1.2, 10, 10]),
    "mediation_ipw_reg_prop_ratio_treatment-M1D_continuous_1DX": np.array([6, 1.2, 1.2, 15, 15]),
    "mediation_ipw_reg_prop_ratio_treatment-M5D_continuous_1DX": np.array([6, 15, 15, 20, 20]),
    "mediation_ipw_reg_prop_ratio_treatment-M5D_continuous_5DX": np.array([2, 5, 5, 10, 10]),
    "mediation_ipw_reg_calibration_prop_ratio_treatment-M1D_binary_1DX": np.array([2, 2, 2, 10, 10]),
    "mediation_ipw_reg_calibration_prop_ratio_treatment-M1D_binary_5DX": np.array([1, 1, 1, 5, 5]),
    "mediation_ipw_reg_calibration_prop_ratio_treatment-M5D_continuous_1DX": np.array([1, 4, 4, 10, 10]),
    "mediation_ipw_reg_calibration_prop_ratio_treatment-M1D_continuous_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_ipw_reg_calibration_prop_ratio_treatment-M5D_continuous_5DX": np.array([1, 6, 6, 15, 15]),
    "mediation_ipw_reg_prop_ratio_mediator-M1D_binary_1DX": np.array([6, 7, 7, 3, 3]),
    "mediation_ipw_reg_prop_ratio_mediator-M1D_binary_5DX": np.array([2, 2, 2, 3, 3]),
    "mediation_ipw_reg_prop_ratio_mediator-M1D_continuous_1DX": np.array([6, 1.2, 1.2, 15, 15]),
    "mediation_ipw_reg_prop_ratio_mediator-M5D_continuous_1DX": np.array([6, 15, 15, 20, 20]),
    "mediation_ipw_reg_prop_ratio_mediator-M5D_continuous_5DX": np.array([2, 5, 5, 10, 10]),
    "mediation_ipw_reg_prop_ratio_treatment_cross_fit-M1D_binary_1DX": np.array([4, 2, 2, 50, 50]),
    "mediation_ipw_reg_prop_ratio_treatment_cross_fit-M1D_binary_5DX": np.array([2, 1.2, 1.2, 10, 10]),
    "mediation_ipw_reg_prop_ratio_treatment_cross_fit-M1D_continuous_1DX": np.array([7, 2, 2, 21, 21]),
    "mediation_ipw_reg_prop_ratio_treatment_cross_fit-M5D_continuous_1DX": np.array([8, 15, 15, 50, 50]),
    "mediation_ipw_reg_prop_ratio_treatment_cross_fit-M5D_continuous_5DX": np.array([2, 5, 5, 10, 10]),
    "mediation_ipw_reg_prop_ratio_treatment_cross_fit-M1D_continuous_5DX": np.array([1, 1, 1, 1.2, 1.2]),
    "mediation_g_computation_reg_integ_implicit-M1D_binary_5DX": np.array([2, 2, 2, 3, 3]),
    "mediation_g_computation_reg_integ_implicit-M1D_continuous_1DX": np.array([1, 1, 1, 1.5, 1.5]),
    "mediation_g_computation_reg_integ_implicit-M5D_continuous_1DX": np.array([1, 1, 1, 1.5, 1.5]),
    "mediation_g_computation_reg_integ_implicit-M1D_continuous_5DX": np.array([2, 2, 2, 4, 4]),
    "mediation_g_computation_reg_integ_implicit-M5D_continuous_5DX": np.array([1, 3, 3, 6, 6]),
    "mediation_g_computation_reg_calibration_integ_implicit-M1D_binary_1DX": np.array([1, 1, 1, 6, 6]),
    "mediation_g_computation_reg_calibration_integ_implicit-M1D_binary_5DX": np.array([1, 1, 1, 5, 5]),
    "mediation_g_computation_reg_calibration_integ_implicit-M1D_continuous_1DX": np.array([2, 2, 2, 4, 4]),
    "mediation_g_computation_reg_calibration_integ_implicit-M5D_continuous_1DX": np.array([1, 2, 2, 5, 5]),
    "mediation_g_computation_reg_calibration_integ_implicit-M5D_continuous_5DX": np.array([6, 15, 15, 20, 20]),
    "mediation_g_computation_reg_integ_implicit_cross_fit-M1D_binary_5DX": np.array([2, 2, 2, 3, 3]),
    "mediation_g_computation_reg_integ_implicit_cross_fit-M1D_continuous_1DX": np.array([1, 1, 1, 1.5, 1.5]),
    "mediation_g_computation_reg_integ_implicit_cross_fit-M5D_continuous_1DX": np.array([1, 1, 1, 1.5, 1.5]),
    "mediation_g_computation_reg_integ_implicit_cross_fit-M1D_continuous_5DX": np.array([2, 2, 2, 4, 4]),
    "mediation_g_computation_reg_integ_implicit_cross_fit-M5D_continuous_5DX": np.array([1, 3, 3, 6, 6]),
    "mediation_g_computation_reg_integ_explicit-M1D_binary_1DX": np.array([3, 1, 1, 50, 50]),
    "mediation_g_computation_reg_integ_explicit-M1D_binary_5DX": np.array([2, 1, 1, 10, 10]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit-M1D_binary_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit-M1D_binary_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit-M1D_continuous_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit-M5D_continuous_1DX": np.array([1, 3, 3, 6, 6]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit-M1D_continuous_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit-M5D_continuous_5DX": np.array([1, 2, 2, 4, 4]),
    "mediation_multiply_robust_reg_calibration_pr_treatment_ig_implicit-M1D_binary_1DX": np.array([1, 1, 1, 3, 3]),
    "mediation_multiply_robust_reg_calibration_pr_treatment_ig_implicit-M1D_binary_5DX": np.array([1, 1, 1, 4, 4]),
    "mediation_multiply_robust_reg_calibration_pr_treatment_ig_implicit-M1D_continuous_1DX": np.array([2, 2, 2, 3, 3]),
    "mediation_multiply_robust_reg_calibration_pr_treatment_ig_implicit-M5D_continuous_1DX": np.array([2, 2, 2, 5, 5]),
    "mediation_multiply_robust_reg_calibration_pr_treatment_ig_implicit-M1D_continuous_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_calibration_pr_treatment_ig_implicit-M5D_continuous_5DX": np.array([1, 3, 3, 4, 4]),
    "mediation_multiply_robust_reg_pr_mediator_ig_implicit-M1D_binary_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_mediator_ig_implicit-M1D_binary_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_mediator_ig_implicit-M1D_continuous_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_mediator_ig_implicit-M5D_continuous_1DX": np.array([1, 3, 3, 6, 6]),
    "mediation_multiply_robust_reg_pr_mediator_ig_implicit-M1D_continuous_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_mediator_ig_implicit-M5D_continuous_5DX": np.array([1, 2, 2, 4, 4]),
    "mediation_multiply_robust_reg_pr_treatment_ig_explicit-M1D_binary_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_explicit-M1D_binary_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_explicit-M1D_continuous_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_explicit-M5D_continuous_1DX": np.array([1, 3, 3, 6, 6]),
    "mediation_multiply_robust_reg_pr_treatment_ig_explicit-M1D_continuous_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_explicit-M5D_continuous_5DX": np.array([1, 2, 2, 4, 4]),
    "mediation_multiply_robust_reg_pr_mediator_ig_explicit-M1D_binary_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_mediator_ig_explicit-M1D_binary_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_mediator_ig_explicit-M1D_continuous_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_mediator_ig_explicit-M5D_continuous_1DX": np.array([1, 3, 3, 6, 6]),
    "mediation_multiply_robust_reg_pr_mediator_ig_explicit-M1D_continuous_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_mediator_ig_explicit-M5D_continuous_5DX": np.array([1, 2, 2, 4, 4]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit_cross_fit-M1D_binary_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit_cross_fit-M1D_binary_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit_cross_fit-M1D_continuous_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit_cross_fit-M5D_continuous_1DX": np.array([1, 3, 3, 6, 6]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit_cross_fit-M1D_continuous_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit_cross_fit-M5D_continuous_5DX": np.array([1, 2, 2, 4, 4]),
    "mediation_dml_reg-M1D_binary_1DX": np.array([1, 2, 2, 6, 6]),
    "mediation_dml_reg-M1D_binary_5DX": np.array([1, 1, 1, 5, 5]),
    "mediation_dml_reg-M5D_continuous_1DX": np.array([1, 10, 10, 20, 20]),
    "mediation_dml_reg-M5D_continuous_5DX": np.array([1, 3, 3, 5, 5]),
    "mediation_dml_reg_calibration-M1D_binary_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_dml_reg_calibration-M1D_continuous_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_dml_reg_calibration-M5D_continuous_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_dml_reg_calibration-M1D_continuous_5DX": np.array([1, 1, 1, 2, 2]),
    "mediation_dml_reg_calibration-M5D_continuous_5DX": np.array([1, 3, 3, 6, 6]),
    "mediation_tmle_prop_ratio_treatment-M1D_binary_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_tmle_prop_ratio_treatment-M1D_continuous_1DX": np.array([1, 1, 1, 2, 2]),
    "mediation_tmle_prop_ratio_treatment-M5D_continuous_1DX": np.array([1, 2, 2, 2, 2]),
    "mediation_tmle_prop_ratio_treatment-M1D_continuous_5DX": np.array([1, 1, 1, 3, 3]),
    "mediation_tmle_prop_ratio_treatment-M5D_continuous_5DX": np.array([3, 3, 3, 15, 15]),
    "mediation_tmle_prop_ratio_mediator-M1D_binary_1DX": np.array([1, 1, 1, 3, 3]),
    "mediation_tmle_prop_ratio_mediator-M1D_binary_5DX": np.array([1, 1, 1, 2, 2]),
}


ESTIMATORS = [
    "coefficient_product",
    "mediation_ipw_reg_prop_ratio_treatment",
    "mediation_ipw_reg_calibration_prop_ratio_treatment",
    "mediation_ipw_reg_prop_ratio_mediator",
    "mediation_ipw_reg_prop_ratio_treatment_cross_fit",
    "mediation_g_computation_reg_integ_implicit",
    "mediation_g_computation_reg_calibration_integ_implicit",
    "mediation_g_computation_reg_integ_explicit",
    "mediation_g_computation_reg_integ_implicit_cross_fit",
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit",
    "mediation_multiply_robust_reg_calibration_pr_treatment_ig_implicit",
    "mediation_multiply_robust_reg_pr_mediator_ig_implicit",
    "mediation_multiply_robust_reg_pr_treatment_ig_explicit",
    "mediation_multiply_robust_reg_pr_mediator_ig_explicit",
    "mediation_multiply_robust_reg_pr_treatment_ig_implicit_cross_fit",
    "mediation_dml_reg",
    "mediation_dml_reg_calibration",
    "mediation_tmle_prop_ratio_treatment",
    "mediation_tmle_prop_ratio_mediator",
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

CONFIGURATION_NAMES = [
    "M1D_binary_1DX",
    "M1D_binary_5DX",
    "M1D_continuous_1DX",
    "M5D_continuous_1DX",
    "M1D_continuous_5DX",
    "M5D_continuous_5DX",
]

CONFIG_DICT = {
    CONFIGURATION_NAMES[i]: dict(zip(PARAMETER_NAME, PARAMETER_LIST[i]))
    for i in range(len(CONFIGURATION_NAMES))
}


ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5
TINY = 1.0e-12
