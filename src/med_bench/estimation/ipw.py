import numpy as np

from med_bench.nuisances.propensities import estimate_treatment_probabilities

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted


class ImportanceWeighting(Estimator):

  def __init__(self, clip : float, trim : float, **kwargs):
    """IPW estimator

    Attributes:
        _clip (float):  clipping the propensities
        _trim (float): remove propensities which are below the trim threshold

    """
    super().__init__(**kwargs)
    self._crossfit = 0
    self._clip = clip
    self._trim = trim

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
        m = m.reshape(n, 1)

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

    self.fit_treatment_propensity_x_nuisance(t, x)
    self.fit_treatment_propensity_xm_nuisance(t, m, x)

    self._fitted = True

    if self.verbose:
        print("Nuisance models fitted")

  @fitted
  def estimate(self, t, m, x, y):
    """Estimates causal effect on data

    """
    t, m, x, y = self.resize(t, m, x, y)
    p_x, p_xm = estimate_treatment_probabilities(t,
                                                  m,
                                                  x,
                                                  self._crossfit,
                                                  self._classifier_t_x,
                                                  self._classifier_t_xm)

    ind = ((p_xm > self._trim) & (p_xm < (1 - self._trim)))
    y, t, p_x, p_xm = y[ind], t[ind], p_x[ind], p_xm[ind]

    # note on the names, ytmt' = Y(t, M(t')), the treatment needs to be
    # binary but not the mediator
    p_x = np.clip(p_x, self._clip, 1 - self._clip)
    p_xm = np.clip(p_xm, self._clip, 1 - self._clip)

    # importance weighting
    y1m1 = np.sum(y * t / p_x) / np.sum(t / p_x)
    y1m0 = np.sum(y * t * (1 - p_xm) / (p_xm * (1 - p_x))) /\
        np.sum(t * (1 - p_xm) / (p_xm * (1 - p_x)))
    y0m0 = np.sum(y * (1 - t) / (1 - p_x)) /\
        np.sum((1 - t) / (1 - p_x))
    y0m1 = np.sum(y * (1 - t) * p_xm / ((1 - p_xm) * p_x)) /\
        np.sum((1 - t) * p_xm / ((1 - p_xm) * p_x))

    total_effect = y1m1 - y0m0
    direct_effect_treated = y1m1 - y0m1
    direct_effect_control = y1m0 - y0m0
    indirect_effect_treated = y1m1 - y1m0
    indirect_effect_control = y0m1 - y0m0

    causal_effects = {
      'total_effect': total_effect,
      'direct_effect_treated': direct_effect_treated,
      'direct_effect_control': direct_effect_control,
      'indirect_effect_treated': indirect_effect_treated,
      'indirect_effect_control': indirect_effect_control
    }

    return causal_effects
