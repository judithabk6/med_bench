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


# CONCLUSION : I can achieve parametrization thanks to a loop and *param,
# yet I'm not sure this is the most elegant solution
