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
import pytest
import numpy as np
import os

from tests.estimation.get_estimation_results import _get_estimation_results
from med_bench.get_simulated_data import simulate_data

from med_bench.utils.utils import DependencyNotInstalledError
from med_bench.utils.constants import (
    CONFIGURATION_NAMES,
    CONFIG_DICT,
    DEFAULT_TOLERANCE,
    TOLERANCE_FACTOR_DICT,
    ESTIMATORS,
)


@pytest.fixture(params=CONFIGURATION_NAMES)
def configuration_name(request):
    return request.param


@pytest.fixture
def dict_param(configuration_name):
    return CONFIG_DICT[configuration_name]


# Two distinct data fixtures
@pytest.fixture
def data_simulated(dict_param):
    return simulate_data(**dict_param)


@pytest.fixture
def x(data_simulated):
    return data_simulated[0]


# t is raveled because some estimators fail with (n,1) inputs
@pytest.fixture
def t(data_simulated):
    return data_simulated[1].ravel()


@pytest.fixture
def m(data_simulated):
    return data_simulated[2]


@pytest.fixture
def y(data_simulated):
    return data_simulated[3].ravel()  # same reason as t


@pytest.fixture
def effects(data_simulated):
    return np.array(data_simulated[4:9])


@pytest.fixture(params=ESTIMATORS)
def estimator(request):
    return request.param


@pytest.fixture
def tolerance(estimator, configuration_name):
    test_name = "{}-{}".format(estimator, configuration_name)
    tolerance = DEFAULT_TOLERANCE
    if test_name in TOLERANCE_FACTOR_DICT.keys():
        tolerance *= TOLERANCE_FACTOR_DICT[test_name]
    return tolerance


@pytest.fixture
def effects_chap(x, t, m, y, estimator):
    # try whether estimator is implemented or not
    try:
        res = _get_estimation_results(x, t, m, y, estimator)[0:5]
    except Exception as e:
        if "1D binary mediator" in str(e):
            pytest.skip(f"{e}")

        else:
            pytest.fail(f"{e}")

    # NaN situations
    if np.all(np.isnan(res)):
        pytest.xfail("all effects are NaN")
    elif np.any(np.isnan(res)):
        pprint("NaN found")

    return res


def test_tolerance(effects, effects_chap, tolerance):
    error = abs((effects_chap - effects) / effects)
    # print(error)
    assert np.all(error[~np.isnan(error)] <= tolerance[~np.isnan(error)])


def test_total_is_direct_plus_indirect(effects_chap):
    if not np.isnan(effects_chap[1]):
        assert effects_chap[0] == pytest.approx(effects_chap[1] + effects_chap[4])
    if not np.isnan(effects_chap[2]):
        assert effects_chap[0] == pytest.approx(effects_chap[2] + effects_chap[3])


def test_robustness_to_ravel_format(data_simulated, estimator, effects_chap):
    if "forest" in estimator:
        pytest.skip("Forest estimator skipped")
    assert np.all(
        _get_estimation_results(
            data_simulated[0],
            data_simulated[1],
            data_simulated[2],
            data_simulated[3],
            estimator,
        )[0:5]
        == pytest.approx(
            effects_chap, nan_ok=True
        )  # effects_chap is obtained with data[1].ravel() and data[3].ravel()
    )
