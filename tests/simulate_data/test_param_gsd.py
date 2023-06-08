"""
    A file testing the ideas of draft_param_gsd
"""

from pprint import pprint
import pytest
import numpy as np
from numpy.random import default_rng
from med_bench.src.get_simulated_data import simulate_data


@pytest.mark.parametrize("n", [1, 2, 100])
@pytest.mark.parametrize("rg", [default_rng(1), default_rng(10)])
@pytest.mark.parametrize("mis_spec_m", [False, True])
@pytest.mark.parametrize("mis_spec_y", [False, True])
@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_m", [1, 2, 3])
@pytest.mark.parametrize("seed", [None, 2, 20])
@pytest.mark.parametrize("type_m", ["binary", "continuous"])
@pytest.mark.parametrize("sigma_y", [0.5, 0.05, 5])
@pytest.mark.parametrize("sigma_m", [0.5, 0.05, 5])
@pytest.mark.parametrize("beta_t_factor", [1, 0.1, 10])
@pytest.mark.parametrize("beta_m_factor", [1, 0.1, 10])
@pytest.fixture(autouse=True)
def effects(
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
    return np.array(data[4:9])


def test_total_is_direct_plus_indirect():
    assert effects[0] == effects[1] + effects[4]  # total = theta_1 + delta_0
    assert effects[0] == effects[2] + effects[3]  # total = theta_0 + delta_1
