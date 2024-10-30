import numpy as np

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import is_array_integer


class GComputation(Estimator):
    """GComputation estimation method class
    """

    def __init__(self, **kwargs):
        """Initalization of the GComputation estimation method class
        """
        super().__init__(**kwargs)

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

        """
        t, m, x, y = self._resize(t, m, x, y)

        self._fit_cross_conditional_mean_outcome_nuisance(t, m, x, y)
        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

        return self

    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        """

        t, m, x, y = self._resize(t, m, x, y)

        mu_0mx, mu_1mx, y0m0, y0m1, y1m0, y1m1 = self._estimate_cross_conditional_mean_outcome_nesting(
            m, x, y)

        # mean score computing
        eta_t1t1 = np.mean(y1m1)
        eta_t0t0 = np.mean(y0m0)
        eta_t1t0 = np.mean(y1m0)
        eta_t0t1 = np.mean(y0m1)

        # effects computing
        total_effect = eta_t1t1 - eta_t0t0

        direct_effect_treated = eta_t1t1 - eta_t0t1
        direct_effect_control = eta_t1t0 - eta_t0t0
        indirect_effect_treated = eta_t1t1 - eta_t1t0
        indirect_effect_control = eta_t0t1 - eta_t0t0

        causal_effects = {
            'total_effect': total_effect,
            'direct_effect_treated': direct_effect_treated,
            'direct_effect_control': direct_effect_control,
            'indirect_effect_treated': indirect_effect_treated,
            'indirect_effect_control': indirect_effect_control
        }

        return causal_effects
