"""
    My Pytest rat lab
"""

from pprint import pprint
import itertools
import pytest
import numpy as np
from numpy.random import default_rng

# export PYTHONPATH="/home/sboumaiz/Bureau/stage_mediation"
from med_bench.src.get_simulated_data import simulate_data


# ATTEMPT TO MERGE PARAMETRIZATION AND TESTS

a = 1


@pytest.mark.parametrize("b", [3, 0, 10], indirect=True)
def fix(b):
    c = b * b
    return c


@pytest.fixture(autouse=True)
def resultat():
    return 3.14 * a


def test_demo():
    assert fix(2) == 4


# def test_demo0():
#     assert a
#     try:
#         print(pi)
#     except:
#         print("An exception occurred")
#     assert resultat() == 3.14


# CHECK WHETHER A LIST IS NEDEED
# use directly itertools, param_list useless?
pamarameter_list = []
for pamarameter in itertools.product(
    [1, 2, 10],
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


for param in pamarameter_list:
    try:
        data = simulate_data(*param)
    except:
        pass
    else:

        def test_dimension_x():
            assert (data[0]).shape == (param[0], param[4])  # (n, dim_x)


# CONCLUSION : Je peux m'en sortir avec une boucle for et *param, mais c'est pas hyper élégant


# # GET HERE SOME SIMULATE_DATA OUTPUT

# data = simulate_data(
#     n=1,
#     rg=default_rng(42),
#     mis_spec_m=False,
#     mis_spec_y=False,
#     dim_x=1,
#     dim_m=1,
#     seed=1,
#     type_m="continuous",
#     sigma_y=0.5,
#     sigma_m=0.5,
#     beta_t_factor=1,
#     beta_m_factor=10000,
# )
# print(data)

# # effects = np.array(data[4:9])
# # print(effects)


# data = []
# try:
#     data = simulate_data(
#         n=3,
#         rg=default_rng(42),
#         mis_spec_m=False,
#         mis_spec_y=False,
#         dim_x=1,
#         dim_m=1,
#         seed=1,
#         type_m="binary",
#         sigma_y=0.5,
#         sigma_m=0.5,
#         beta_t_factor=10,
#         beta_m_factor=1,
#     )
# except ValueError as err:
#     print(err)
# else:
#     print(data)
