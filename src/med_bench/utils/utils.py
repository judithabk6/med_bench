import numpy as np

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri, numpy2ri

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
            variables[index] = var.reshape(-1,1)
    pre_inter_variables = np.hstack(variables)
    if not interaction:
        return pre_inter_variables
    new_cols = list()
    for i, var in enumerate(variables[:]):
        for j, var2 in enumerate(variables[i+1:]):
            for ii in range(var.shape[1]):
                for jj in range(var2.shape[1]):
                    new_cols.append((var[:, ii] * var2[:, jj]).reshape(-1, 1))
    new_vars = np.hstack(new_cols)
    result = np.hstack((pre_inter_variables, new_vars))
    return result

def _convert_array_to_R(x):
    """
    converts a numpy array to a R matrix or vector
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.sum(a == np.array(_convert_array_to_R(a)))
    6
    """
    if len(x.shape) == 1:
        return robjects.FloatVector(x)
    elif len(x.shape) == 2:
        return robjects.r.matrix(robjects.FloatVector(x.ravel()),
                                 nrow=x.shape[0], byrow='TRUE')
