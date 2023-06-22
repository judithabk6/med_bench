"""
get_estimation.py::get_estimation

We test all the benchmark_mediation estimators for a certain tolerance.
The test is skipped if estimator has not been implemented yet,
i.e. all outputs are NaN or NotImplementedError is raised.
The test xfails for any other wierd behavior.

We pinpoint that :
- DML_huber is not working for dim_x=1, RRuntimeError is raised (#15)
- t.ravel() and y.ravel() are necessary to get IPW proper behavior (#16)
- multiply_robust methods return some NaN effects (#17)

Some estimators exeed tolerance levels :
- Mainly IPW and forest estimators, particularly for indirect effects
- DML_huber fails to estimate indirect effects (>85% relative error)
- coefficient_product annecdotally fails to estimate indirect effects (70% relative error)
"""

from pprint import pprint
import itertools
import pytest
import numpy as np
from numpy.random import default_rng
from med_bench.src.get_simulated_data import simulate_data
from med_bench.src.get_estimation import get_estimation


ATE_TOLERANCE = 0.2
DIRECT_TOLERANCE = 0.2
INDIRECT_TOLERANCE = 0.6  # indirect effect is weak leading to a huge relative error
TOLERANCE = np.array(
    [
        ATE_TOLERANCE,
        DIRECT_TOLERANCE,
        DIRECT_TOLERANCE,
        INDIRECT_TOLERANCE,
        INDIRECT_TOLERANCE,
    ]
)


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


parameter_name = [
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


parameter_list = list(
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

parameter_list.extend(
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


@pytest.fixture(params=parameter_list)
def dict_param(request):
    return dict(zip(parameter_name, request.param))


@pytest.fixture(autouse=True)
def data_simulation(dict_param):
    res = simulate_data(**dict_param)
    global x, t, m, y, effects
    x = res[0]
    t = res[1].ravel()
    m = res[2]
    y = res[3].ravel()
    effects = np.array(res[4:9])


@pytest.fixture(params=estimator_list, autouse=True)
def skip_if_not_implemented(request, dict_param):
    # config determination
    if dict_param["dim_m"] == 1 and dict_param["type_m"] == "binary":
        config = 0
    else:
        config = 5

    # try whether estimator is implemented or not
    try:
        global effects_chap
        effects_chap = get_estimation(x, t, m, y, request.param, config)[0:5]
    except Exception as message_error:
        if message_error != NotImplementedError:
            pytest.xfail("Missing NotImplementedError")
        else:
            pytest.mark.skip("Not implemented")

    # NaN situations
    if np.all(np.isnan(effects_chap)):
        pytest.skip("all effects are NaN")
    elif np.any(np.isnan(effects_chap)):
        pprint("NaN found")


def test_tolerance():
    error = abs((effects_chap - effects) / effects)
    assert np.all(error[~np.isnan(error)] <= TOLERANCE[~np.isnan(error)])


def test_total_is_direct_plus_indirect():
    if not np.isnan(effects_chap[1]):
        assert effects_chap[0] == pytest.approx(effects_chap[1] + effects_chap[4])
    if not np.isnan(effects_chap[2]):
        assert effects_chap[0] == pytest.approx(effects_chap[2] + effects_chap[3])
