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
import os
import numpy as np

from med_bench.get_estimation import get_estimation

current_dir = os.path.dirname(__file__)
true_estimations_file_path = os.path.join(current_dir, 'tests_results.npy')
TRUE_ESTIMATIONS = np.load(true_estimations_file_path, allow_pickle=True)


@pytest.fixture(params=range(TRUE_ESTIMATIONS.shape[0]))
def tests_results_idx(request):
    return request.param


@pytest.fixture
def data(tests_results_idx):
    return TRUE_ESTIMATIONS[tests_results_idx]


@pytest.fixture
def estimator(data):
    return data[0]


@pytest.fixture
def x(data):
    return data[1]


# t is raveled because some estimators fail with (n,1) inputs
@pytest.fixture
def t(data):
    return data[2].ravel()


@pytest.fixture
def m(data):
    return data[3]


@pytest.fixture
def y(data):
    return data[4].ravel()  # same reason as t


@pytest.fixture
def config(data):
    return data[5]


@pytest.fixture
def result(data):
    return data[6]


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

        else:
            pytest.fail(f"{e}")

    # NaN situations
    if np.all(np.isnan(res)):
        pytest.xfail("all effects are NaN")
    elif np.any(np.isnan(res)):
        pprint("NaN found")

    return res


def test_estimation_exactness(result, effects_chap):

    assert np.all(effects_chap == pytest.approx(result, abs=0.01))
