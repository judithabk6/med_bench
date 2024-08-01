import numpy as np

from sklearn.linear_model import RidgeCV

from med_bench.estimation.base import Estimator
from med_bench.utils.constants import ALPHAS, CV_FOLDS, TINY
from med_bench.utils.decorators import fitted


class CoefficientProduct(Estimator):

    def __init__(self, clip : float, trim : float, regularize : bool, **kwargs):
        """Coefficient product estimator

        Attributes:
            clip (float):  clipping the propensities
            trim (float): remove propensities which are below the trim threshold
            regularize (bool) : regularization parameter

        """
        super().__init__(**kwargs)
        self._crossfit = 0
        self._regularize = regularize
        self._clip = clip
        self._trim = trim 

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
        self.fit_score_nuisances(t, m, x, y)
        # estimate mediator densities

        if self._regularize:
            alphas = ALPHAS
        else:
            alphas = [TINY]
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(m.shape) == 1:
            m = m.reshape(-1, 1)
        if len(t.shape) == 1:
            t = t.reshape(-1, 1)

        self._coef_t_m = np.zeros(m.shape[1])
        for i in range(m.shape[1]):
            m_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS) \
                .fit(np.hstack((x, t)), m[:, i])
            self._coef_t_m[i] = m_reg.coef_[-1]
        y_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS) \
            .fit(np.hstack((x, t, m)), y.ravel())

        self._coef_y = y_reg.coef_

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")


    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        """
        direct_effect_treated = self._coef_y[x.shape[1]]
        direct_effect_control = direct_effect_treated
        indirect_effect_treated = sum(self._coef_y[x.shape[1] + 1:] * self._coef_t_m)
        indirect_effect_control = indirect_effect_treated

        causal_effects = {
            'total_effect': direct_effect_treated+indirect_effect_control,
            'direct_effect_treated': direct_effect_treated,
            'direct_effect_control': direct_effect_control,
            'indirect_effect_treated': indirect_effect_treated,
            'indirect_effect_control': indirect_effect_control
        }
        return causal_effects
