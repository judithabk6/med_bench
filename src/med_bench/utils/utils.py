from numpy.random import default_rng
import numpy as np
import pandas as pd
import subprocess

from med_bench.get_simulated_data import simulate_data
from med_bench.utils.constants import ALPHAS, TINY



def _get_interactions(interaction, *args):
    """
    this function provides interaction terms between different groups of
    variables (confounders, treatment, mediators)

    Parameters
    ----------
    interaction : boolean
                    whether to compute interaction terms

    *args : flexible, one or several arrays
                    blocks of variables between which interactions should be
                    computed


    Returns
    --------
    array_like
        interaction terms

    Examples
    --------
    >>> x = np.arange(6).reshape(3, 2)
    >>> t = np.ones((3, 1))
    >>> m = 2 * np.ones((3, 1))
    >>> get_interactions(False, x, t, m)
    array([[0., 1., 1., 2.],
           [2., 3., 1., 2.],
           [4., 5., 1., 2.]])
    >>> get_interactions(True, x, t, m)
    array([[ 0.,  1.,  1.,  2.,  0.,  1.,  0.,  2.,  2.],
           [ 2.,  3.,  1.,  2.,  2.,  3.,  4.,  6.,  2.],
           [ 4.,  5.,  1.,  2.,  4.,  5.,  8., 10.,  2.]])
    """
    variables = list(args)
    for index, var in enumerate(variables):
        if len(var.shape) == 1:
            variables[index] = var.reshape(-1, 1)
    pre_inter_variables = np.hstack(variables)
    if not interaction:
        return pre_inter_variables
    new_cols = list()
    for i, var in enumerate(variables[:]):
        for j, var2 in enumerate(variables[i + 1:]):
            for ii in range(var.shape[1]):
                for jj in range(var2.shape[1]):
                    new_cols.append((var[:, ii] * var2[:, jj]).reshape(-1, 1))
    new_vars = np.hstack(new_cols)
    result = np.hstack((pre_inter_variables, new_vars))
    return result


class DependencyNotInstalledError(Exception):
    pass


def _check_input(y, t, m, x, setting):
    """
    internal function to check inputs. `_check_input` adjusts the dimension
    of the input (matrix or vectors), and raises an error
    - if the size of input is not adequate,
    - or if the type of input is not supported (cotinuous treatment or
    non-binary one-dimensional mediator if the specified setting parameter
    is binary)

    Parameters
    ----------
    y : array-like, shape (n_samples)
        Outcome value for each unit, continuous

    t : array-like, shape (n_samples)
        Treatment value for each unit, binary

    m : array-like, shape (n_samples, n_mediators)
        Mediator value for each unit, binary and unidimensional

    x : array-like, shape (n_samples, n_features_covariates)
        Covariates value for each unit, continuous

    setting : string
    ('binary', 'continuous', 'multidimensional') value for the mediator

    Returns
    -------
    y_converted : array-like, shape (n_samples,)
        Outcome value for each unit, continuous

    t_converted : array-like, shape (n_samples,)
        Treatment value for each unit, binary

    m_converted : array-like, shape (n_samples, n_mediators)
        Mediator value for each unit, binary and unidimensional

    x_converted : array-like, shape (n_samples, n_features_covariates)
        Covariates value for each unit, continuous
    """
    # check format
    if len(y) != len(y.ravel()):
        raise ValueError("Multidimensional y (outcome) is not supported")

    if len(t) != len(t.ravel()):
        raise ValueError("Multidimensional t (exposure) is not supported")

    if len(np.unique(t)) != 2:
        raise ValueError("Only a binary t (exposure) is supported")

    n = len(y)
    t_converted = t.ravel()
    y_converted = y.ravel()

    if n != len(x) or n != len(m) or n != len(t):
        raise ValueError("Inputs don't have the same number of observations")

    if len(x.shape) == 1:
        x_converted = x.reshape(n, 1)
    else:
        x_converted = x

    if len(m.shape) == 1:
        m_converted = m.reshape(n, 1)
    else:
        m_converted = m

    if (m_converted.shape[1] > 1) and (setting != "multidimensional"):
        raise ValueError("Multidimensional m (mediator) is not supported")

    if (setting == "binary") and (len(np.unique(m)) != 2):
        raise ValueError(
            "Only a binary one-dimensional m (mediator) is supported")

    return y_converted, t_converted, m_converted, x_converted


def is_array_integer(array):
    if array.shape[1] > 1:
        return False
    return all(list((array == array.astype(int)).squeeze()))


def is_array_binary(array):
    if len(np.unique(array)) == 2:
        return True
    return False


def _get_regularization_parameters(regularization):
    """
    Obtain regularization parameters

    Returns
    -------
    cs : list
        each of the values in Cs describes the inverse of regularization
        strength for predictors
    alphas : list
        alpha values to try in ridge models
    """
    if regularization:
        alphas = ALPHAS
        cs = ALPHAS
    else:
        alphas = [TINY]
        cs = [np.inf]

    return cs, alphas
