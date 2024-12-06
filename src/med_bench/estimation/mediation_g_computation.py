import numpy as np

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import is_array_binary


class GComputation(Estimator):
    """GComputation estimation method class"""

    def __init__(self, regressor, classifier, **kwargs):
        """Initializes GComputation estimation method

        Parameters
        ----------
        regressor
            Regressor used for mu estimation, can be any object with a fit and predict method
        classifier
            Classifier used for propensity estimation, can be any object with a fit and predict_proba method
        """
        super().__init__(**kwargs)

        assert hasattr(
            regressor, 'fit'), "The model does not have a 'fit' method."
        assert hasattr(
            regressor, 'predict'), "The model does not have a 'predict' method."
        assert hasattr(
            classifier, 'fit'), "The model does not have a 'fit' method."
        assert hasattr(
            classifier, 'predict_proba'), "The model does not have a 'predict_proba' method."
        self.regressor = regressor
        self.classifier = classifier

    def _estimate_conditional_mean_outcome_table(self, x):
        """
        Estimate conditional mean outcome E[Y|T,M,X]

        Returns
        -------
        mu_00x: array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=0,M=0,X]
        mu_01x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=0,M=1,X]
        mu_10x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=1,M=0,X]
        mu_11x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=1,M=1,X]
        """
        n = x.shape[0]

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))
        m0 = np.zeros((n, 1))
        m1 = np.ones((n, 1))

        x_t1_m1 = np.hstack([x, t1.reshape(-1, 1), m1])
        x_t1_m0 = np.hstack([x, t1.reshape(-1, 1), m0])
        x_t0_m1 = np.hstack([x, t0.reshape(-1, 1), m1])
        x_t0_m0 = np.hstack([x, t0.reshape(-1, 1), m0])

        mu_00x = self._regressor_y.predict(x_t0_m0)
        mu_01x = self._regressor_y.predict(x_t0_m1)
        mu_10x = self._regressor_y.predict(x_t1_m0)
        mu_11x = self._regressor_y.predict(x_t1_m1)

        return mu_00x, mu_01x, mu_10x, mu_11x

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data"""
        t, m, x, y = self._resize(t, m, x, y)

        if is_array_binary(m):
            self._fit_binary_mediator_probability(t, m, x)
            self._fit_conditional_mean_outcome(t, m, x, y)
        else:
            self._fit_cross_conditional_mean_outcome(t, m, x, y)

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

        return self

    @ fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        """
        t, m, x, y = self._resize(t, m, x, y)

        if is_array_binary(m):
            f_00x, f_01x, f_10x, f_11x = \
                self._estimate_binary_mediator_probability_table(x)
            mu_00x, mu_01x, mu_10x, mu_11x = \
                self._estimate_conditional_mean_outcome_table(x)

            direct_effect_i1 = mu_11x - mu_01x
            direct_effect_i0 = mu_10x - mu_00x
            n = len(y)
            direct_effect_treated = (direct_effect_i1 * f_11x
                                     + direct_effect_i0 * f_10x).sum() / n
            direct_effect_control = (direct_effect_i1 * f_01x
                                     + direct_effect_i0 * f_00x).sum() / n
            indirect_effect_i1 = f_11x - f_01x
            indirect_effect_i0 = f_10x - f_00x
            indirect_effect_treated = (indirect_effect_i1 * mu_11x
                                       + indirect_effect_i0 * mu_10x).sum() / n
            indirect_effect_control = (indirect_effect_i1 * mu_01x
                                       + indirect_effect_i0 * mu_00x).sum() / n
            total_effect = direct_effect_control + indirect_effect_treated
        else:
            (mu_0mx, mu_1mx, y0m0, y0m1, y1m0, y1m1) = \
                self._estimate_cross_conditional_mean_outcome(m, x)

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
            "total_effect": total_effect,
            "direct_effect_treated": direct_effect_treated,
            "direct_effect_control": direct_effect_control,
            "indirect_effect_treated": indirect_effect_treated,
            "indirect_effect_control": indirect_effect_control,
        }

        return causal_effects
