
import numpy as np

from med_bench.get_simulated_data import simulate_data
from med_bench.get_estimation import get_estimation

from med_bench.utils.constants import ESTIMATORS, PARAMETER_LIST, PARAMETER_NAME


def get_data_from_list(data):
    x = data[0]
    t = data[1].ravel()
    m = data[2]
    y = data[3].ravel()

    return x, t, m, y


def get_config_from_dict(dict_params):
    if dict_params["dim_m"] == 1 and dict_params["type_m"] == "binary":
        config = 0
    else:
        config = 5
    return config


def get_estimators_results(x, t, m, y, config, estimator):

    try:
        res = get_estimation(x, t, m, y, estimator, config)[0:5]
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
        x, t, m, y = get_data_from_list(data)
        config = get_config_from_dict(dict_params=dict_params)

        for estimator in ESTIMATORS:

            # Get results from synthetic inputs
            result = get_estimators_results(x, t, m, y, config, estimator)
            row = [estimator, x, t, m, y, config, result]
            results.append(row)

    # Store the results in a npy file
    np.save("tests_results.npy", np.array(results, dtype="object"))
