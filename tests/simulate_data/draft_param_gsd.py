"""
    A draft articulating the test_get_simulated_data parametrization
"""


from pprint import pprint
import itertools
import pytest
import numpy as np
from numpy.random import default_rng
from med_bench.get_simulated_data import simulate_data


# Could we directly use itertools, parameter_list may be a duplicate?
parameter_list = []
for parameter in itertools.product(
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
    parameter_list.append(parameter)


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
    parameter_list,
)
class ParametrizedTest:
    def test_total_is_direct_plus_indirect(
        self,
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
        effects = np.array(data[4:9])

        assert effects[0] == effects[1] + effects[4]  # total = theta_1 + delta_0
        assert effects[0] == effects[2] + effects[3]  # total = theta_0 + delta_1

    def test_dimension_x(n, dim_x):
        assert (data[0]).shape == (n, dim_x)



