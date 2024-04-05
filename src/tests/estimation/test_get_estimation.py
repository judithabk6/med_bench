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

from med_bench.get_simulated_data import simulate_data
from med_bench.get_estimation import get_estimation

from med_bench.utils.utils import DependencyNotInstalledError, check_r_dependencies
from med_bench.utils.constants import PARAMETER_LIST, PARAMETER_NAME, R_DEPENDENT_ESTIMATORS, TOLERANCE_DICT


@pytest.fixture(params=PARAMETER_LIST)
def dict_param(request):
    return dict(zip(PARAMETER_NAME, request.param))


@pytest.fixture
def data(dict_param):
    return simulate_data(**dict_param)


@pytest.fixture
def x(data):
    return data[0]


# t is raveled because some estimators fail with (n,1) inputs
@pytest.fixture
def t(data):
    return data[1].ravel()


@pytest.fixture
def m(data):
    return data[2]


@pytest.fixture
def y(data):
    return data[3].ravel()  # same reason as t


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
    except Exception as e:
        if str(e) in (
            "Estimator only supports 1D binary mediator.",
            "Estimator does not support 1D binary mediator.",
        ):
            pytest.skip(f"{e}")

        # We skip the test if an error with function from glmet rpy2 package occurs
        elif "glmnet::glmnet" in str(e):
            pytest.skip(f"{e}")

        elif estimator in R_DEPENDENT_ESTIMATORS and not check_r_dependencies():
            assert isinstance(e, DependencyNotInstalledError) == True
            return e

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
    assert np.all(error[~np.isnan(error)] <= tolerance[~np.isnan(error)])


def test_total_is_direct_plus_indirect(effects_chap):
    if not np.isnan(effects_chap[1]):
        assert effects_chap[0] == pytest.approx(
            effects_chap[1] + effects_chap[4])
    if not np.isnan(effects_chap[2]):
        assert effects_chap[0] == pytest.approx(
            effects_chap[2] + effects_chap[3])


@pytest.mark.xfail
def test_robustness_to_ravel_format(data, estimator, config, effects_chap):
    if "forest" in estimator:
        pytest.skip("Forest estimator skipped")
    if not isinstance(effects_chap, DependencyNotInstalledError):
        assert np.all(
            get_estimation(data[0], data[1], data[2],
                           data[3], estimator, config)[0:5]
            == pytest.approx(
                effects_chap, nan_ok=True
            )  # effects_chap is obtained with data[1].ravel() and data[3].ravel()
        )
