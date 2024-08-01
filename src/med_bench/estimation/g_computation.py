import numpy as np

from sklearn.cluster import KMeans

from med_bench.estimation.base import Estimator

from med_bench.utils.decorators import fitted
from med_bench.utils.utils import (is_array_integer)

from med_bench.nuisances.density import estimate_mediators_probabilities
from med_bench.nuisances.conditional_outcome import estimate_conditional_mean_outcome
from med_bench.nuisances.cross_conditional_outcome import estimate_cross_conditional_mean_outcome_nesting

class GComputation(Estimator):
    """GComputation estimation method class
    """
    def __init__(self, crossfit : int, procedure : str, **kwargs):
        """Initalization of the GComputation estimation method class

        Parameters
        ----------
        crossfit : int
            1 or 0 
        procedure : str
            nesting or discrete
        """
        super().__init__(**kwargs)

        self._crossfit = crossfit
        self._procedure = procedure

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

        return t, m, x, y
    

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

        """

        self.fit_score_nuisances(t, m, x, y)
        t, m, x, y = self.resize(t, m, x, y)
        
        if self._procedure == 'discrete':
            if not is_array_integer(m):
                self._bucketizer = KMeans(n_clusters=10, random_state=self.rng,
                                n_init="auto").fit(m)
                m = self._bucketizer.predict(m)
            self.fit_mediator_nuisance(t, m, x)
            self.fit_conditional_mean_outcome_nuisance(t, m, x, y)

        elif self._procedure == 'nesting':
            self.fit_cross_conditional_mean_outcome_nuisance(t, m, x, y)
        else:
            raise NotImplementedError
        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

    def estimate_discrete(self, t, m, x, y):
        """Estimates causal effect on data using a discrete summation on
        mediators

        """
        
        # estimate mediator densities
        f_t0, f_t1 = estimate_mediators_probabilities(t, m, x, y,
                                                            self._crossfit,
                                                            self._classifier_m,
                                                            False)

        # estimate conditional mean outcomes
        mu_t0, mu_t1, _, _ = (
            estimate_conditional_mean_outcome(t, m, x, y,
                                                        self._crossfit,
                                                        self._regressor_y,
                                                        False))

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
    
    def estimate_nesting(self, t, m, x, y):
            
        mu_0mx, mu_1mx, y0m0, y0m1, y1m0, y1m1 = estimate_cross_conditional_mean_outcome_nesting(m, x, y, self.regressors)

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

        t, m, x, y = self.resize(t, m, x, y)

        if self._procedure == 'discrete':
            if not is_array_integer(m):
                m = self._bucketizer.predict(m)
            return self.estimate_discrete(t, m, x, y)
        
        elif self._procedure == 'nesting':
            return self.estimate_nesting(t, m, x, y)


