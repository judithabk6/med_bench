import numpy as np
from sklearn.cluster import KMeans

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import (is_array_integer)
from med_bench.nuisances.cross_conditional_outcome import estimate_cross_conditional_mean_outcome_discrete, estimate_cross_conditional_mean_outcome_nesting
from med_bench.nuisances.propensities import estimate_treatment_propensity_x, estimate_treatment_probabilities
from med_bench.nuisances.density import estimate_mediator_density, estimate_mediators_probabilities


class MultiplyRobust(Estimator):
    """Implementation of multiply robust

    Args:
        settings (dict): dictionnary of parameters
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, procedure : str, ratio : str, density_estimation_method : str, clip : float, normalized, **kwargs):
        super().__init__(**kwargs)

        self._crossfit = 0
        self._procedure = procedure
        self._ratio = ratio
        self._density_estimation = density_estimation_method
        self._clip = clip
        self._normalized = normalized

    def resize(self, t, m, x, y):
        """Resize data for the right shape

        Parameters
        ----------
        t       array-like, shape (n_samples)
                treatment value for each unit, binary

        m       array-like, shape (n_samples)
                mediator value for each unit, here m is necessary binary and uni-
                dimensional

        x       array-like, shape (n_samples, n_features_covariates)
                covariates (potential confounders) values

        y       array-like, shape (n_samples)
                outcome value for each unit, continuous
        """
        if len(y) != len(y.ravel()):
            raise ValueError("Multidimensional y is not supported")
        if len(t) != len(t.ravel()):
            raise ValueError("Multidimensional t is not supported")

        n = len(y)
        if len(x.shape) == 1:
            x.reshape(n, 1)
        if len(m.shape) == 1:
            m.reshape(n, 1)

        if n != len(x) or n != len(m) or n != len(t):
            raise ValueError(
                "Inputs don't have the same number of observations")

        y = y.ravel()
        t = t.ravel()
        # m = m.ravel()

        return t, m, x, y


    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

        """
        # bucketize if needed
        t, m, x, y = self.resize(t, m, x, y)

        # fit nuisance functions
        self.fit_score_nuisances(t, m, x, y)

        if self._ratio == 'density':
            self.fit_treatment_propensity_x_nuisance(t, x)
            self.fit_mediator_nuisance(t, m, x)

        elif self._ratio == 'propensities':
            self.fit_treatment_propensity_x_nuisance(t, x)
            self.fit_treatment_propensity_xm_nuisance(t, m, x)

        else:
            raise NotImplementedError
        
        if self._procedure == 'nesting':
            self.fit_cross_conditional_mean_outcome_nuisance(t, m, x, y)

        elif self._procedure == 'discrete':

            if not is_array_integer(m):
                self._bucketizer = KMeans(n_clusters=10, random_state=self.rng,
                                          n_init="auto").fit(m)
                m = self._bucketizer.predict(m)
            self.fit_mediator_nuisance(t, m, x)
            self.fit_cross_conditional_mean_outcome_nuisance_discrete(t, m, x, y)
        
        else:
            raise NotImplementedError

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")


    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        """
        # Format checking
        t, m, x, y = self.resize(t, m, x, y)
            
        if self._ratio == 'density':
            f_m0x, f_m1x = estimate_mediator_density(t, m, x, y,
                                                                self._crossfit,
                                                                self._classifier_m,
                                                                False)
            p_x = estimate_treatment_propensity_x(t,
                                                       m,
                                                       x,
                                                       self._crossfit,
                                                       self._classifier_t_x)
            ratio_t1_m0 = f_m0x / (p_x * f_m1x)
            ratio_t0_m1 = f_m1x / ((1 - p_x) * f_m0x)

        elif self._ratio == 'propensities':
            p_x, p_xm = estimate_treatment_probabilities(t,
                                                    m,
                                                    x,
                                                    self._crossfit,
                                                    self._classifier_t_x,
                                                    self._classifier_t_xm)
            ratio_t1_m0 = (1-p_xm) / ((1 - p_x) * p_xm)
            ratio_t0_m1 = p_xm / ((1 - p_xm) * p_x)

        if self._procedure == 'nesting':
            

            # p_x = estimate_treatment_propensity_x(t,
            #                                         m,
            #                                         x,
            #                                         self._crossfit,
            #                                         self._classifier_t_x)

            # _, _, f_m0x, f_m1x = estimate_mediator_density(t,
            #                                                                 m,
            #                                                                 x,
            #                                                                 y,
            #                                                         self._crossfit,
            #                                                         self._classifier_m,
            #                                                         False)

            mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1 = (
                estimate_cross_conditional_mean_outcome_nesting(m, x, y, self.regressors))


        else:
            if not is_array_integer(m):
                m = self._bucketizer.predict(m)

            # p_x = estimate_treatment_propensity_x(t,
            #                                        m,
            #                                        x,
            #                                        self._crossfit,
            #                                        self._classifier_t_x)

            f_t0, f_t1 = estimate_mediators_probabilities(t,
                                                                           m,
                                                                           x,
                                                                           y,
                                                                   self._crossfit,
                                                                   self._classifier_m,
                                                                   False)

            f = f_t0, f_t1
            mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1 = (
                estimate_cross_conditional_mean_outcome_discrete(m, x, y, f, self.regressors))


        # clipping
        # p_x_clip = p_x != np.clip(p_x, self._clip, 1 - self._clip)
        # f_m0x_clip = f_m0x != np.clip(f_m0x, self._clip, 1 - self._clip)
        # f_m1x_clip = f_m1x != np.clip(f_m1x, self._clip, 1 - self._clip)
        # clipped = p_x_clip + f_m0x_clip + f_m1x_clip

        # var_name = ["t", "y", "p_x", "f_m0x", "f_m1x", "mu_1mx", "mu_0mx"]
        # var_name += ["E_mu_t1_t1", "E_mu_t0_t0", "E_mu_t1_t0", "E_mu_t0_t1"]
        # n_discarded = 0
        # for var in var_name:
        #     exec(f"{var} = {var}[~clipped]")
        # n_discarded += np.sum(clipped)

        # score computing
        if self._normalized:
            sum_score_m1 = np.mean(t / p_x)
            sum_score_m0 = np.mean((1 - t) / (1 - p_x))
            # sum_score_t1m0 = np.mean((t / p_x) * (f_m0x / f_m1x))
            # sum_score_t0m1 = np.mean((1 - t) / (1 - p_x) * (f_m1x / f_m0x))
            sum_score_t1m0 = np.mean(t * ratio_t1_m0)
            sum_score_t0m1 = np.mean((1 - t) * ratio_t0_m1)

            y1m1 = (t / p_x * (y - E_mu_t1_t1)) / sum_score_m1 + E_mu_t1_t1
            y0m0 = (((1 - t) / (1 - p_x) * (y - E_mu_t0_t0)) / sum_score_m0
                    + E_mu_t0_t0)
            # y1m0 = (
            #         ((t / p_x) * (f_m0x / f_m1x) * (
            #                     y - mu_1mx)) / sum_score_t1m0
            #         + ((1 - t) / (1 - p_x) * (
            #             mu_1mx - E_mu_t1_t0)) / sum_score_m0
            #         + E_mu_t1_t0
            # )
            y1m0 = (
                    (t * ratio_t1_m0 * (
                                y - mu_1mx)) / sum_score_t1m0
                    + ((1 - t) / (1 - p_x) * (
                        mu_1mx - E_mu_t1_t0)) / sum_score_m0
                    + E_mu_t1_t0
            )
            # y0m1 = (
            #         ((1 - t) / (1 - p_x) * (f_m1x / f_m0x) * (y - mu_0mx))
            #         / sum_score_t0m1 + t / p_x * (
            #                 mu_0mx - E_mu_t0_t1) / sum_score_m1
            #         + E_mu_t0_t1
            # )
            y0m1 = (
                    ((1 - t) * ratio_t0_m1 * (y - mu_0mx))
                    / sum_score_t0m1 + t / p_x * (
                            mu_0mx - E_mu_t0_t1) / sum_score_m1
                    + E_mu_t0_t1
            )
        else:
            y1m1 = t / p_x * (y - E_mu_t1_t1) + E_mu_t1_t1
            y0m0 = (1 - t) / (1 - p_x) * (y - E_mu_t0_t0) + E_mu_t0_t0
            # y1m0 = (
            #         (t / p_x) * (f_m0x / f_m1x) * (y - mu_1mx)
            #         + (1 - t) / (1 - p_x) * (mu_1mx - E_mu_t1_t0)
            #         + E_mu_t1_t0
            # )
            # y0m1 = (
            #         (1 - t) / (1 - p_x) * (f_m1x / f_m0x) * (y - mu_0mx)
            #         + t / p_x * (mu_0mx - E_mu_t0_t1)
            #         + E_mu_t0_t1
            # )
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
