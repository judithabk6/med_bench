from med_bench.estimation.ipw import ImportanceWeighting
from med_bench.estimation.g_computation import GComputation
from med_bench.estimation.dml import DoubleMachineLearning
from med_bench.estimation.mr import MultiplyRobust
from med_bench.estimation.tmle import TMLE


def get_estimator_by_name(settings):
    if settings['estimator'] == 'ipw':
        return ImportanceWeighting
    elif settings['estimator'] == 'g_computation':
        return GComputation
    elif settings['estimator'] == 'linear':
        return Linear
    elif settings['estimator'] == 'mr':
        return MultiplyRobust
    elif settings['estimator'] == 'dml':
        return DoubleMachineLearning
    elif settings['estimator'] == 'tmle':
        return TMLE
    else:
        raise NotImplementedError