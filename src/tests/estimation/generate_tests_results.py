import numpy as np

from med_bench.get_simulated_data import simulate_data
from tests.estimation.get_estimation_results import _get_estimation_results

from med_bench.utils.constants import ESTIMATORS, PARAMETER_LIST, PARAMETER_NAME


def _get_data_from_list(data):
    """Get x, t, m, y from simulated data
    """
    x = data[0]
    t = data[1].ravel()
    m = data[2]
    y = data[3].ravel()

    return x, t, m, y


def _get_config_from_dict(dict_params):
    """Get config parameter used for estimators parametrisation
    """
    if dict_params["dim_m"] == 1 and dict_params["type_m"] == "binary":
        config = 0
    else:
        config = 5
    return config


def _get_estimators_results(x, t, m, y, config, estimator):
    """Get estimation result from specified input parameters and estimator name
    """

    try:
        res = _get_estimation_results(x, t, m, y, estimator, config)[0:5]
        return res

    except Exception as e:
        print(f"{e}")
        return str(e)


if __name__ == "__main__":

    results = []

    for param_list in PARAMETER_LIST:

        # Get synthetic input data from parameters list defined above
        dict_params = dict(zip(PARAMETER_NAME, param_list))
        data = simulate_data(**dict_params)
        x, t, m, y = _get_data_from_list(data)
        config = _get_config_from_dict(dict_params=dict_params)

        for estimator in ESTIMATORS:

            # Get results from synthetic inputs
            result = _get_estimators_results(x, t, m, y, config, estimator)
            row = [estimator, x, t, m, y, config, result]
            results.append(row)

    # Store the results in a npy file
    np.save("tests_results.npy", np.array(results, dtype="object"))
