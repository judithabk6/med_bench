import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from scipy.special import logit

from med_bench.estimation.base import Estimator
from med_bench.utils.constants import CV_FOLDS
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import _get_regularization_parameters


class CoefficientProduct(Estimator):
    """Coefficient Product estimatation method class"""

    def __init__(self, regularize: bool, **kwargs):
        """Initializes Coefficient product estimatation method

        Parameters
        ----------
            regularize (bool) : regularization parameter
        """
        super().__init__(**kwargs)

        self._regularize = regularize
        self.mediator_cardinality_threshold = 2

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

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
        cs, alphas = _get_regularization_parameters(regularization=self._regularize)

        t, m, x, y = self._resize(t, m, x, y)
        self._fit_mediator_discretizer(m)
        if self._mediator_considered_discrete:
            self.classifier = LogisticRegressionCV(random_state=42, 
                                                   Cs=cs,
                                                   cv=CV_FOLDS)
            m_label, m_discrete_value = self._discretize_mediators(m)
            self._fit_discrete_mediator_probability(t, m_label, x)
        else:
            self._coef_t_m = np.zeros(m.shape[1])
            for i in range(m.shape[1]):
                m_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(
                    np.hstack((x, t.reshape(-1, 1))), m[:, i]
                )
                self._coef_t_m[i] = m_reg.coef_[-1]
        y_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(
            np.hstack((x, t.reshape(-1, 1), m)), y
        )

        self._coef_y = y_reg.coef_

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

    @fitted
    def estimate(self, t, m, x, y):
        t, m, x, y = self._resize(t, m, x, y)
        """Estimates causal effect on data"""
        direct_effect_treated = self._coef_y[x.shape[1]]
        direct_effect_control = direct_effect_treated
        if self._mediator_considered_discrete:
            f_0x, f_1x = self._estimate_discrete_mediator_probability_table(x)
            indirect_effect_treated = (self._coef_y[x.shape[1] + 1] * \
                (f_1x[1] - f_0x[1])).mean()
            indirect_effect_control = indirect_effect_treated
        else:
            indirect_effect_treated = sum(
                self._coef_y[x.shape[1] + 1 :] * self._coef_t_m
            )
            indirect_effect_control = indirect_effect_treated

        causal_effects = {
            "total_effect": direct_effect_treated + indirect_effect_control,
            "direct_effect_treated": direct_effect_treated,
            "direct_effect_control": direct_effect_control,
            "indirect_effect_treated": indirect_effect_treated,
            "indirect_effect_control": indirect_effect_control,
            "total_effect_variance": None,
            "direct_effect_treated_variance": None,
            "direct_effect_control_variance": None,
            "indirect_effect_treated_variance": None,
            "indirect_effect_control_variance": None,
        }
        return causal_effects
