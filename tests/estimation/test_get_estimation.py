"""
Pytest file for get_estimation.py

It tests all the benchmark_mediation estimators :
- for a certain tolerance
- whether their effects satisfy "total = direct + indirect"
- whether they support (n,1) and (n,) inputs

To be robust to future updates, tests are adjusted with a smaller tolerance when possible.
The test is skipped if estimator has not been implemented yet, i.e. if ValueError is raised.
The test fails for any other unwanted behavior.
"""

from pprint import pprint
import itertools
import pytest
import numpy as np
from numpy.random import default_rng
from med_bench.src.get_simulated_data import simulate_data
from med_bench.src.get_estimation import get_estimation


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
    "huber_ipw_noreg": INFINITE_TOLERANCE,
    "huber_ipw_reg": INFINITE_TOLERANCE,
    "huber_ipw_reg_calibration": INFINITE_TOLERANCE,
    "huber_ipw_forest": INFINITE_TOLERANCE,
    "huber_ipw_forest_calibration": INFINITE_TOLERANCE,
    "g_computation_noreg": LARGE_TOLERANCE,
    "g_computation_reg": MEDIUM_TOLERANCE,
    "g_computation_reg_calibration": LARGE_TOLERANCE,
    "g_computation_forest": LARGE_TOLERANCE,
    "g_computation_forest_calibration": INFINITE_TOLERANCE,
    "multiply_robust_noreg": MEDIUM_TOLERANCE,
    "multiply_robust_reg": SMALL_TOLERANCE,
    "multiply_robust_reg_calibration": SMALL_TOLERANCE,
    "multiply_robust_forest": INFINITE_TOLERANCE,
    "multiply_robust_forest_calibration": LARGE_TOLERANCE,
    "simulation_based": LARGE_TOLERANCE,
    "DML_huber": INFINITE_TOLERANCE,
    "G_estimator": SMALL_TOLERANCE,
    "huber_ipw_noreg_cf": INFINITE_TOLERANCE,
    "huber_ipw_reg_cf": INFINITE_TOLERANCE,
    "huber_ipw_reg_calibration_cf": INFINITE_TOLERANCE,
    "huber_ipw_forest_cf": INFINITE_TOLERANCE,
    "huber_ipw_forest_calibration_cf": INFINITE_TOLERANCE,
    "g_computation_noreg_cf": SMALL_TOLERANCE,
    "g_computation_reg_cf": LARGE_TOLERANCE,
    "g_computation_reg_calibration_cf": LARGE_TOLERANCE,
    "g_computation_forest_cf": INFINITE_TOLERANCE,
    "g_computation_forest_calibration_cf": LARGE_TOLERANCE,
    "multiply_robust_noreg_cf": MEDIUM_TOLERANCE,
    "multiply_robust_reg_cf": SMALL_TOLERANCE,
    "multiply_robust_reg_calibration_cf": MEDIUM_TOLERANCE,
    "multiply_robust_forest_cf": INFINITE_TOLERANCE,
    "multiply_robust_forest_calibration_cf": MEDIUM_TOLERANCE,
}


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


@pytest.fixture(params=PARAMETER_LIST)
def dict_param(request):
    return dict(zip(PARAMETER_NAME, request.param))


@pytest.fixture
def data(dict_param):
    return simulate_data(**dict_param)


@pytest.fixture
def x(data):
    return data[0]


@pytest.fixture
def t(data):
    return data[1].ravel()


@pytest.fixture
def m(data):
    return data[2]


@pytest.fixture
def y(data):
    return data[3].ravel()


@pytest.fixture
def effects(data):
    return np.array(data[4:9])


@pytest.fixture(params=list(TOLERANCE_DICT.keys()))
def estimator(request):
    return request.param


@pytest.fixture
def tolerance(estimator):
    return TOLERANCE_DICT[estimator]


@pytest.fixture
def config(dict_param):
    if dict_param["dim_m"] == 1 and dict_param["type_m"] == "binary":
        return 0
    return 5


@pytest.fixture
def effects_chap(x, t, m, y, estimator, config):
    # try whether estimator is implemented or not
    try:
        res = get_estimation(x, t, m, y, estimator, config)[0:5]
    except ValueError as message_error:
        if message_error.args[0] in (
            "Estimator only supports 1D binary mediator.",
            "Estimator does not support 1D binary mediator.",
        ):
            pytest.skip(f"{message_error}")
        else:
            pytest.fail(f"{message_error}")

    # NaN situations
    if np.all(np.isnan(res)):
        pytest.xfail("all effects are NaN")
    elif np.any(np.isnan(res)):
        pprint("NaN found")

    return res


def test_tolerance(effects, effects_chap, tolerance):
    error = abs((effects_chap - effects) / effects)
    assert np.all(error[~np.isnan(error)] <= tolerance[~np.isnan(error)])


def test_total_is_direct_plus_indirect(effects_chap):
    if not np.isnan(effects_chap[1]):
        assert effects_chap[0] == pytest.approx(effects_chap[1] + effects_chap[4])
    if not np.isnan(effects_chap[2]):
        assert effects_chap[0] == pytest.approx(effects_chap[2] + effects_chap[3])


@pytest.mark.xfail
def test_robustness_to_ravel_format(data, estimator, config, effects_chap):
    if "forest" in estimator:
        pytest.skip("Forest estimator skipped")
    assert np.all(
        get_estimation(data[0], data[1], data[2], data[3], estimator, config)[0:5]
        == pytest.approx(effects_chap, nan_ok=True)
    )
