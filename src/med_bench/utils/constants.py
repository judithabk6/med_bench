import itertools
import numpy as np
from numpy.random import default_rng

# CONSTANTS USED FOR TESTS

# TOLERANCE THRESHOLDS

TOLERANCE_THRESHOLDS = {
    "SMALL": {
        "ATE": 0.05,
        "DIRECT": 0.05,
        "INDIRECT": 0.2,
    },
    "MEDIUM": {
        "ATE": 0.10,
        "DIRECT": 0.10,
        "INDIRECT": 0.4,
    },
    "LARGE": {
        "ATE": 0.15,
        "DIRECT": 0.15,
        "INDIRECT": 0.9,
    },
    "INFINITE": {
        "ATE": np.inf,
        "DIRECT": np.inf,
        "INDIRECT": np.inf,
    },
}


def get_tolerance_array(tolerance_size: str) -> np.array:
    """Get tolerance array for tolerance tests

    Parameters
    ----------
    tolerance_size : str
        tolerance size, can be "SMALL", "MEDIUM", "LARGE" or "INFINITE"

    Returns
    -------
    np.array
        array of size 5 containing the ATE, DIRECT (*2) and INDIRECT (*2) effects tolerance
    """

    return np.array(
        [
            TOLERANCE_THRESHOLDS[tolerance_size]["ATE"],
            TOLERANCE_THRESHOLDS[tolerance_size]["DIRECT"],
            TOLERANCE_THRESHOLDS[tolerance_size]["DIRECT"],
            TOLERANCE_THRESHOLDS[tolerance_size]["INDIRECT"],
            TOLERANCE_THRESHOLDS[tolerance_size]["INDIRECT"],
        ]
    )


SMALL_TOLERANCE = get_tolerance_array("SMALL")
MEDIUM_TOLERANCE = get_tolerance_array("MEDIUM")
LARGE_TOLERANCE = get_tolerance_array("LARGE")
INFINITE_TOLERANCE = get_tolerance_array("INFINITE")

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
    "mediation_dml": INFINITE_TOLERANCE,
    "mediation_dml_noreg": INFINITE_TOLERANCE,
    "mediation_dml_reg": INFINITE_TOLERANCE,
    "mediation_dml_forest": INFINITE_TOLERANCE,
    "mediation_g_estimator": LARGE_TOLERANCE,
}

ESTIMATORS = list(TOLERANCE_DICT.keys())

R_DEPENDENT_ESTIMATORS = [
    "mediation_IPW_R",
    "simulation_based",
    "mediation_dml",
    "mediation_g_estimator",
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

ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5
TINY = 1.0e-12
