from med_bench.estimation.mediation_ipw import ImportanceWeighting
from med_bench.estimation.mediation_g_computation import GComputation
from med_bench.estimation.mediation_coefficient_product import CoefficientProduct
from med_bench.estimation.mediation_dml import DoubleMachineLearning
from med_bench.estimation.mediation_mr import MultiplyRobust
from med_bench.estimation.mediation_tmle import TMLE


def get_estimator_by_name(settings):
    if settings['estimator'] == 'ipw':
        return ImportanceWeighting
    elif settings['estimator'] == 'g_computation':
        return GComputation
    elif settings['estimator'] == 'coefficient_product':
        return CoefficientProduct
    elif settings['estimator'] == 'mr':
        return MultiplyRobust
    elif settings['estimator'] == 'dml':
        return DoubleMachineLearning
    elif settings['estimator'] == 'tmle':
        return TMLE
    else:
        raise NotImplementedError
