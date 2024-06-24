import pytest
import re
import numpy as np
from numpy.random import default_rng
from med_bench.get_simulated_data import simulate_data
from med_bench.utils.utils import _check_input


@pytest.fixture
def data():
    return simulate_data(100,
                         default_rng(321),
                         mis_spec_m=False,
                         mis_spec_y=False,
                         dim_x=5,
                         dim_m=1,
                         seed=None,
                         type_m='binary',
                         sigma_y=0.5,
                         sigma_m=0.5,
                         beta_t_factor=1,
                         beta_m_factor=1)


@pytest.fixture
def x(data):
    return data[0]


@pytest.fixture
def t(data):
    return data[1]


@pytest.fixture
def m(data):
    return data[2]


@pytest.fixture
def y(data):
    return data[3].ravel()  # same reason as t


def test_dim_input(y, t, m, x):
    with pytest.raises(ValueError,
                       match=re.escape("Multidimensional y (outcome)")):
        _check_input(x, t, m, x, 'binary')
    with pytest.raises(ValueError,
                       match=re.escape("Multidimensional t (exposure)")):
        _check_input(y, x, m, x, 'binary')
    with pytest.raises(ValueError,
                       match=re.escape("Only a binary t (exposure)")):
        _check_input(y, x[:, 0], m, x, 'binary')
    with pytest.raises(ValueError,
                       match=re.escape("same number of observations")):
        _check_input(y, np.vstack([t, t]), m, x, 'binary')
    with pytest.raises(ValueError,
                       match=re.escape("same number of observations")):
        _check_input(y, t, np.vstack([m, m]), x, 'binary')
    with pytest.raises(ValueError,
                       match=re.escape("same number of observations")):
        _check_input(y, t, m, np.vstack([x, x]), 'binary')


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


def test_m_type(y, t, m, x):
    with pytest.raises(ValueError,
                       match=re.escape("Multidimensional m (mediator)")):
        _check_input(y, t, x, x, 'binary')
    with pytest.raises(ValueError,
                       match=re.escape("a binary one-dimensional m")):
        _check_input(y, t, x[:, 0], x, 'binary')
