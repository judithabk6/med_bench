from numpy.random import default_rng
import numpy as np
import pandas as pd
import subprocess

from med_bench.get_simulated_data import simulate_data
from med_bench.utils.constants import ALPHAS, TINY


def check_r_dependencies():
    try:
        # Check if R is accessible by trying to get its version
        subprocess.check_output(["R", "--version"])

        # If the above command fails, it will raise a subprocess.CalledProcessError and won't reach here

        # Assuming reaching here means R is accessible, now try importing rpy2 packages
        import rpy2.robjects.packages as rpackages

        required_packages = [
            "causalweight",
            "mediation",
            "stats",
            "base",
            "grf",
            "plmed",
        ]

        for package in required_packages:
            rpackages.importr(package)

        return True  # All checks passed, R and required packages are available

    except:
        # Handle the case where R is not found or rpy2 is not installed
        return False


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


def is_r_installed():
    try:
        subprocess.check_output(["R", "--version"])
        return True
    except:
        return False


def check_r_package(package_name):
    try:
        import rpy2.robjects.packages as rpackages

        rpackages.importr(package_name)
        return True
    except:
        return False


class DependencyNotInstalledError(Exception):
    pass


def r_dependency_required(required_packages):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_r_installed():
                raise DependencyNotInstalledError(
                    "R is not installed or not found. "
                    "Please install R and set it up correctly in your system."
                )

            # To get rid of the 'DataFrame' object has no attribute 'iteritems' error due to pandas version mismatch in rpy2
            # https://stackoverflow.com/a/76404841
            pd.DataFrame.iteritems = pd.DataFrame.items

            for package in required_packages:
                if not check_r_package(package):
                    if package != "plmed":
                        raise DependencyNotInstalledError(
                            f"The '{package}' R package is not installed. "
                            "Please install it using R by running:\n"
                            "import rpy2.robjects.packages as rpackages\n"
                            "utils = rpackages.importr('utils')\n"
                            "utils.chooseCRANmirror(ind=33)\n"
                            f"utils.install_packages('{package}')"
                        )
                    else:
                        raise DependencyNotInstalledError(
                            "The 'plmed' R package is not installed. "
                            "Please install it using R by running:\n"
                            "import rpy2.robjects.packages as rpackages\n"
                            "utils = rpackages.importr('utils')\n"
                            "utils.chooseCRANmirror(ind=33)\n"
                            "utils.install_packages('devtools')\n"
                            "devtools = rpackages.importr('devtools')\n"
                            "devtools.install_github('ohines/plmed')"
                        )
                    return None
            return func(*args, **kwargs)

        return wrapper

    return decorator


if is_r_installed():
    import rpy2.robjects as robjects


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
        return robjects.r.matrix(
            robjects.FloatVector(x.ravel()), nrow=x.shape[0], byrow="TRUE"
        )


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
