"""
benchmark_mediation.py::ols_mediation

We test :
- Coefficient product estimator for a certain tolerance
- Whether total effect = direct + indirect
"""

from pprint import pprint
import pytest
import numpy as np
from numpy.random import default_rng
from med_bench.src.get_simulated_data import simulate_data
from med_bench.src.benchmark_mediation import ols_mediation


ATE_TOLERANCE = 0.05
DIRECT_TOELERANCE = 0.05
INDIRECT_TOLERANCE = 0.05
TOLERANCE = [
    ATE_TOLERANCE,
    DIRECT_TOELERANCE,
    DIRECT_TOELERANCE,
    INDIRECT_TOLERANCE,
    INDIRECT_TOLERANCE,
]


data = simulate_data(
    n=1000,
    rg=default_rng(43),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=1,
    seed=1,
    type_m="binary",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=1,
    beta_m_factor=1,
)
x = data[0]
t = data[1]
m = data[2]
y = data[3]
effects = np.array(data[4:9])


@pytest.mark.parametrize("interaction", [False, True])
@pytest.mark.parametrize("regularization", [False, True])
class TestParametrizedOls:
    @pytest.mark.parametrize("error_tolerance", [TOLERANCE])
    def test_tolerance(self, interaction, regularization, error_tolerance):
        effects_chap = ols_mediation(y, t, m, x, interaction, regularization)
        effects_chap = effects_chap[0:5]
        error = abs((effects - effects_chap) / effects)
        assert np.all(error <= error_tolerance)

    def test_total_is_direct_plus_indirect(self, interaction, regularization):
        effects_chap = ols_mediation(y, t, m, x, interaction, regularization)
        assert effects_chap[0] == effects_chap[1] + effects_chap[4]
        assert effects_chap[0] == effects_chap[2] + effects_chap[3]
