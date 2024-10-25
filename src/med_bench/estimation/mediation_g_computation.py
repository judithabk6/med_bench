import numpy as np

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import is_array_integer


class GComputation(Estimator):
    """GComputation estimation method class
    """

    def __init__(self, crossfit: int, procedure: str, **kwargs):
        """Initalization of the GComputation estimation method class

        Parameters
        ----------
        crossfit : int
            Any integer value greater than 0.
        procedure : str
            nesting or discrete
        """
        super().__init__(**kwargs)

        self._crossfit = crossfit
        self._procedure = procedure

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

        """

        self._fit_nuisance(t, m, x, y)
        t, m, x, y = self._resize(t, m, x, y)

        if self._procedure == 'nesting':
            self._fit_cross_conditional_mean_outcome_nuisance(t, m, x, y)
        else:
            raise NotImplementedError
        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

        return self

    def _estimate_discrete(self, t, m, x, y):
        """Estimates causal effect on data using a discrete summation on
        mediators

        """

        # estimate mediator densities
        f_t0, f_t1 = self._estimate_mediators_probabilities(t, m, x, y)

        # estimate conditional mean outcomes
        mu_t0, mu_t1, _, _ = (
            self._estimate_conditional_mean_outcome(t, m, x, y))

        n = len(y)

        direct_effect_treated = 0
        direct_effect_control = 0
        indirect_effect_treated = 0
        indirect_effect_control = 0

        for f_1bx, f_0bx, mu_1bx, mu_0bx in zip(f_t1, f_t0, mu_t1, mu_t0):
            direct_effect_ib = mu_1bx - mu_0bx
            direct_effect_treated += direct_effect_ib * f_1bx
            direct_effect_control += direct_effect_ib * f_0bx
            indirect_effect_ib = f_1bx - f_0bx
            indirect_effect_treated += indirect_effect_ib * mu_1bx
            indirect_effect_control += indirect_effect_ib * mu_0bx

        direct_effect_treated = direct_effect_treated.sum() / n
        direct_effect_control = direct_effect_control.sum() / n
        indirect_effect_treated = indirect_effect_treated.sum() / n
        indirect_effect_control = indirect_effect_control.sum() / n

        total_effect = direct_effect_control + indirect_effect_treated

        causal_effects = {
            'total_effect': total_effect,
            'direct_effect_treated': direct_effect_treated,
            'direct_effect_control': direct_effect_control,
            'indirect_effect_treated': indirect_effect_treated,
            'indirect_effect_control': indirect_effect_control
        }
        return causal_effects

    def _estimate_nesting(self, t, m, x, y):

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

    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        """

        t, m, x, y = self._resize(t, m, x, y)

        if self._procedure == 'discrete':
            if not is_array_integer(m):
                m = self._bucketizer.predict(m)
            return self._estimate_discrete(t, m, x, y)

        elif self._procedure == 'nesting':
            return self._estimate_nesting(t, m, x, y)
