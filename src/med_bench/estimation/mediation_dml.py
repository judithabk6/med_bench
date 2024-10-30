import numpy as np

from med_bench.estimation.base import Estimator


class DoubleMachineLearning(Estimator):
    """Implementation of double machine learning

    Parameters
    ----------
        alpha (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * alpha then vector is discarded
    """

    def __init__(self, clip: float, trim: float, normalized: bool, **kwargs):
        super().__init__(**kwargs)

        self._clip = clip
        self._trim = trim
        self._normalized = normalized

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

        """
        t, m, x, y = self._resize(t, m, x, y)

        self._fit_treatment_propensity_x_nuisance(t, x)
        self._fit_treatment_propensity_xm_nuisance(t, m, x)
        self._fit_cross_conditional_mean_outcome_nuisance(t, m, x, y)
        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        """
        t, m, x, y = self._resize(t, m, x, y)

        p_x, p_xm = self._estimate_treatment_probabilities(t, m, x)

        mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1 = self._estimate_cross_conditional_mean_outcome_nesting(
            m, x, y)

        not_trimmed = (
            (((1 - p_xm) * p_x) >= self._trim)
            * ((1 - p_x) >= self._trim)
            * (p_x >= self._trim)
            * ((p_xm * (1 - p_x)) >= self._trim)
        )

        var_name = [
            "p_x",
            "p_xm",
            "mu_1mx",
            "mu_0mx",
            "E_mu_t1_t0",
            "E_mu_t0_t1",
            "E_mu_t1_t1",
            "E_mu_t0_t0",
        ]
        for var in var_name:
            exec(f"{var} = {var}[not_trimmed]")
        nobs = np.sum(not_trimmed)

        # score computing
        if self._normalized:
            sum_score_m1 = np.mean(t / p_x)
            sum_score_m0 = np.mean((1 - t) / (1 - p_x))
            sum_score_t1m0 = np.mean(t * (1 - p_xm) / (p_xm * (1 - p_x)))
            sum_score_t0m1 = np.mean((1 - t) * p_xm / ((1 - p_xm) * p_x))
            y1m1 = (t / p_x * (y - E_mu_t1_t1)) / sum_score_m1 + E_mu_t1_t1
            y0m0 = (((1 - t) / (1 - p_x) * (y - E_mu_t0_t0)) / sum_score_m0
                    + E_mu_t0_t0)
            y1m0 = (
                (t * (1 - p_xm) / (p_xm * (1 - p_x)) * (y - mu_1mx))
                / sum_score_t1m0 + (
                    (1 - t) / (1 - p_x) * (mu_1mx - E_mu_t1_t0))
                / sum_score_m0 + E_mu_t1_t0
            )
            y0m1 = (
                ((1 - t) * p_xm / ((1 - p_xm) * p_x) * (y - mu_0mx))
                / sum_score_t0m1
                + (t / p_x * (mu_0mx - E_mu_t0_t1)) / sum_score_m1
                + E_mu_t0_t1
            )
        else:
            y1m1 = t / p_x * (y - E_mu_t1_t1) + E_mu_t1_t1
            y0m0 = (1 - t) / (1 - p_x) * (y - E_mu_t0_t0) + E_mu_t0_t0
            y1m0 = (
                t * (1 - p_xm) / (p_xm * (1 - p_x)) * (y - mu_1mx)
                + (1 - t) / (1 - p_x) * (mu_1mx - E_mu_t1_t0)
                + E_mu_t1_t0
            )
            y0m1 = (
                (1 - t) * p_xm / ((1 - p_xm) * p_x) * (y - mu_0mx)
                + t / p_x * (mu_0mx - E_mu_t0_t1)
                + E_mu_t0_t1
            )

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
