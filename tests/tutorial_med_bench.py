"""Simulation de données et application de quelques estimateurs

-E
huber_piw_noreg : Effet indirect erroné, peu importe la graine
simulate_data : Dimension 5 en binaire non implémenté

-W
simulate data : La graine a peu d'effet sur l'ordre de grandeur des effets
"""

from pprint import pprint
from numpy.random import default_rng
import numpy as np


# environment variable to import judith functions
# export PYTHONPATH="/home/sboumaiz/Bureau/stage_mediation"
from judith_abecassis.src.get_simulated_data_new import simulate_data

# Errors from test_simulation_settings_new :
# from rpy2.rinterface_lib.embedded import RRuntimeError
# installed twangMediation
# commented estimator test part because of "folderpath"
from judith_abecassis.src.test_simulation_settings_new import get_estimation

# from judith_abecassis.src.benchmark_mediation.src.benchmark_mediation import huber_IPW


# simulate data & estimator application
estimator_list = ["coefficient_product", "huber_ipw_reg", "DML_huber"]
SAMPLE_SIZE = 1000
CONFIG = 1

data = simulate_data(
    n=SAMPLE_SIZE,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=5,
    dim_m=1,
    seed=1,
    type_m="binary",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=10,
    beta_m_factor=1,
)

print(data)
effects = np.array(data[4:9])
print(effects)
est_effects = []
err = []

for estimator in estimator_list:
    res = get_estimation(data[0], data[1], data[2], data[3], estimator, CONFIG)
    est_effects.append(res)
    effects_chap = res[0:5]
    err.append((effects - effects_chap) / effects)

pprint(effects)
pprint(est_effects)
pprint(err)
