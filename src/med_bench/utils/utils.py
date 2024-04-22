import numpy as np
import pandas as pd

import subprocess
import warnings


def check_r_dependencies():
    try:
        # Check if R is accessible by trying to get its version
        subprocess.check_output(["R", "--version"])

        # If the above command fails, it will raise a subprocess.CalledProcessError and won't reach here

        # Assuming reaching here means R is accessible, now try importing rpy2 packages
        import rpy2.robjects.packages as rpackages
        required_packages = [
            'causalweight', 'mediation', 'stats', 'base', 'grf', 'plmed'
        ]

        for package in required_packages:
            rpackages.importr(package)

        return True  # All checks passed, R and required packages are available

    except:
        # Handle the case where R is not found or rpy2 is not installed
        return False


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
                    if package != 'plmed':
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
