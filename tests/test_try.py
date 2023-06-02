import pytest

a = 1


@pytest.mark.parametrize("b", [3, 0, 10], indirect=True)
def fix(b):
    c = b * b
    return c


@pytest.fixture(autouse=True)
def resultat():
    pass


def test_demo():
    assert d
    # try:
    #     print(a)
    # except:
    #     print("An exception occurred")
    # assert resultat(1) == 0
    pass


from pprint import pprint
import itertools
import pytest
import numpy as np
from numpy.random import default_rng
from .get_simulated_data import simulate_data


# @pytest.fixture(autouse=True)
# def data():
#     return simulate_data(10, default_rng(7))


# @pytest.fixture(autouse=True)
# def effects():
#     return np.array(data[4:9])


@pytest.mark.parametrize.fixture(autouse=True)
def fix():
    pass


pamarameter_list = []
for pamarameter in itertools.product(
    [1, 2, 100],
    [default_rng(1), default_rng(10)],
    [False, True],
    [False, True],
    [1, 2, 3],
    [1, 2, 3],
    [None, 2, 20],
    ["binary", "continuous"],
    [0.5, 0.05, 5],
    [0.5, 0.05, 5],
    [1, 0.1, 10],
    [1, 0.1, 10],
):
    pamarameter_list.append(pamarameter)
# use directly itertools, param_list useless?


data = None
err_returned = True


@pytest.fixture(autouse=True)
def setup(
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
):
    try:
        err_returned = False
        data = simulate_data(
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
    except ValueError as err:
        err_returned = True


@pytest.mark.parametrize(
    [
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
    ],
    pamarameter_list,
)
class ParametrizedTest:
    def test_total_is_direct_plus_indirect(self):
        assert effects[0] == effects[1] + effects[4]  # total = theta_1 + delta_0
        assert effects[0] == effects[2] + effects[3]  # total = theta_0 + delta_1

    def test_effects_are_equals_if_y_well_specified(self):
        pass
