import numpy as np

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import is_array_integer


class MultiplyRobust(Estimator):
    """Implementation of multiply robust estimator
    """

    def __init__(self, ratio: str, clip: float, normalized, **kwargs):
        super().__init__(**kwargs)

        assert ratio in ['density', 'propensities']
        self._ratio = ratio
        self._clip = clip
        self._normalized = normalized

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data
        """
        t, m, x, y = self._resize(t, m, x, y)

        # fit nuisance functions
        self._fit_nuisance(t, m, x, y)

        if self._ratio == 'density' and is_array_integer(m):
            self._fit_treatment_propensity_x_nuisance(t, x)
            self._fit_mediator_nuisance(t, m, x)

        elif self._ratio == 'propensities':
            self._fit_treatment_propensity_x_nuisance(t, x)
            self._fit_treatment_propensity_xm_nuisance(t, m, x)

        elif self._ratio == 'density' and not is_array_integer(m):
            raise NotImplementedError("""Continuous mediator cannot use the density ratio method, 
                                      use a discrete mediator or set the ratio to 'propensities'""")

        self._fit_cross_conditional_mean_outcome_nuisance(t, m, x, y)

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

        return self

    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        """
        # Format checking
        t, m, x, y = self.resize(t, m, x, y)

        if self._ratio == 'density':
            f_m0x, f_m1x = self._estimate_mediator_probability(t, m, x, y)
            p_x = self._estimate_treatment_propensity_x(t, m, x)
            ratio_t1_m0 = f_m0x / (p_x * f_m1x)
            ratio_t0_m1 = f_m1x / ((1 - p_x) * f_m0x)

        elif self._ratio == 'propensities':
            p_x, p_xm = self._estimate_treatment_probabilities(t, m, x)
            ratio_t1_m0 = (1-p_xm) / ((1 - p_x) * p_xm)
            ratio_t0_m1 = p_xm / ((1 - p_xm) * p_x)

        mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1 = (
            self._estimate_cross_conditional_mean_outcome_nesting(m, x, y))

        # score computing
        if self._normalized:
            sum_score_m1 = np.mean(t / p_x)
            sum_score_m0 = np.mean((1 - t) / (1 - p_x))
            sum_score_t1m0 = np.mean(t * ratio_t1_m0)
            sum_score_t0m1 = np.mean((1 - t) * ratio_t0_m1)

            y1m1 = (t / p_x * (y - E_mu_t1_t1)) / sum_score_m1 + E_mu_t1_t1
            y0m0 = (((1 - t) / (1 - p_x) * (y - E_mu_t0_t0)) / sum_score_m0
                    + E_mu_t0_t0)

            y1m0 = (
                (t * ratio_t1_m0 * (
                    y - mu_1mx)) / sum_score_t1m0
                + ((1 - t) / (1 - p_x) * (
                    mu_1mx - E_mu_t1_t0)) / sum_score_m0
                + E_mu_t1_t0
            )

            y0m1 = (
                ((1 - t) * ratio_t0_m1 * (y - mu_0mx))
                / sum_score_t0m1 + t / p_x * (
                    mu_0mx - E_mu_t0_t1) / sum_score_m1
                + E_mu_t0_t1
            )
        else:
            y1m1 = t / p_x * (y - E_mu_t1_t1) + E_mu_t1_t1
            y0m0 = (1 - t) / (1 - p_x) * (y - E_mu_t0_t0) + E_mu_t0_t0

            y1m0 = (
                t * ratio_t1_m0 * (y - mu_1mx)
                + (1 - t) / (1 - p_x) * (mu_1mx - E_mu_t1_t0)
                + E_mu_t1_t0
            )
            y0m1 = (
                (1 - t) * ratio_t0_m1 * (y - mu_0mx)
                + t / p_x * (mu_0mx - E_mu_t0_t1)
                + E_mu_t0_t1
            )

        # effects computing
        total = np.mean(y1m1 - y0m0)
        direct1 = np.mean(y1m1 - y0m1)
        direct0 = np.mean(y1m0 - y0m0)
        indirect1 = np.mean(y1m1 - y1m0)
        indirect0 = np.mean(y0m1 - y0m0)

        causal_effects = {
            'total_effect': total,
            'direct_effect_treated': direct1,
            'direct_effect_control': direct0,
            'indirect_effect_treated': indirect1,
            'indirect_effect_control': indirect0
        }
        return causal_effects
