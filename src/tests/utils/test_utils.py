import pytest
import re
import numpy as np
from numpy.random import default_rng
from scipy.special import expit

from med_bench.get_simulated_data import simulate_data
from med_bench.utils.utils import _check_input


rg = default_rng(5)
n = 5
dim_x = 3

x = rg.standard_normal(n * dim_x).reshape((n, dim_x))
binary_m_or_t = rg.binomial(1, 0.5, n).reshape(-1, 1)
y = rg.standard_normal(n).reshape(-1, 1)


testdata = [
    (x, binary_m_or_t, binary_m_or_t, x, "Multidimensional y (outcome)"),
    (y, x, binary_m_or_t, x, "Multidimensional t (exposure)"),
    (y, x[0], binary_m_or_t, x, "Only a binary t (exposure)"),
    (y, np.vstack([binary_m_or_t, binary_m_or_t]), binary_m_or_t, x, 
        "same number of observations"),
    (y, binary_m_or_t, np.vstack([binary_m_or_t, binary_m_or_t]), x, 
        "same number of observations"),
    (y, binary_m_or_t, binary_m_or_t, np.vstack([x, x]),
        "same number of observations"),
    (y, binary_m_or_t, x, x, "Multidimensional m (mediator)"),
    (y, binary_m_or_t, x[:, 0], x, "a binary one-dimensional m"),
    ]
ids = ['outcome dimension',
       'exposure dimension',
       'continuous exposure',
       'number of observations (t)',
       'number of observations (m)',
       'number of observations (x)',
       'mediator dimension',
       'binary mediator']

@pytest.mark.parametrize("y, t, m, x, match", testdata, ids=ids)
def test_dim_input(y, t, m, x, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        _check_input(y, t, m, x, 'binary')

@pytest.mark.parametrize("y, t, m, x", [(y, binary_m_or_t, binary_m_or_t, x)])
def test_dim_output(y, t, m, x):
    n = len(y)
    y_converted, t_converted, m_converted, x_converted = \
        _check_input(y, t, m, x, 'binary')
    assert y_converted.shape == (n,)
    assert t_converted.shape == (n,)
    assert x_converted.shape == x.shape
    assert m_converted.shape == m.shape
    y_converted, t_converted, m_converted, x_converted = \
        _check_input(y, t, m.ravel(), x[:, 0], 'binary')
    assert x_converted.shape == (n, 1)
    assert m_converted.shape == (n, 1)


