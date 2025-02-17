import numpy as np

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted


class InversePropensityWeighting(Estimator):
    """Inverse propensity weighting estimation method class"""

    def __init__(
        self, classifier, clip: float, trim: float, prop_ratio="treatment", **kwargs
    ):
        """Initializes Inverse propensity weighting estimation method

        Parameters
        ----------
        classifier
            Classifier used for propensity estimation, can be any object with a fit and predict_proba method
        clip : float
            Clipping value for propensity scores
        trim : float
            Trimming value for propensity scores
        prop_ratio : str
            propensities ratio to use for estimation, can be either 'mediator' or 'treatment'
        """
        super().__init__(**kwargs)

        assert hasattr(classifier, "fit"), "The model does not have a 'fit' method."
        assert hasattr(
            classifier, "predict_proba"
        ), "The model does not have a 'predict_proba' method."
        self.classifier = classifier

        self._clip = clip
        self._trim = trim
        self._prop_ratio = prop_ratio
        self.name = "IPW"

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data"""

        t, m, x, y = self._resize(t, m, x, y)
        self._fit_treatment_propensity_x(t, x)

        if self._prop_ratio == "treatment":
            self._fit_treatment_propensity_xm(t, m, x)
        elif self._prop_ratio == "mediator":
            self._fit_mediator_probability(t, m, x)

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

        return self

    def _pointwise_estimate(self, t, m, x, y):
        """Point wise estimate of the causal effect on data"""

        t, m, x, y = self._resize(t, m, x, y)
        p_x = self._estimate_treatment_propensity_x(x)
        p_x = np.clip(p_x, self._clip, 1 - self._clip)

        if self._prop_ratio == "treatment":
            p_xm = self._estimate_treatment_propensity_xm(m, x)
            p_xm = np.clip(p_xm, self._clip, 1 - self._clip)
            prop_ratio_t1_m0 = (1 - p_xm) / ((1 - p_x) * p_xm)
            prop_ratio_t0_m1 = p_xm / ((1 - p_xm) * p_x)

        elif self._prop_ratio == "mediator":
            f_m0x, f_m1x = self._estimate_mediator_probability(x, m)
            f_m0x = np.clip(f_m0x, self._clip, None)
            f_m1x = np.clip(f_m1x, self._clip, None)
            prop_ratio_t1_m0 = f_m0x / (p_x * f_m1x)
            prop_ratio_t0_m1 = f_m1x / ((1 - p_x) * f_m0x)

        ind = (p_x > self._trim) & (p_x < (1 - self._trim))
        y, t, p_x, prop_ratio_t1_m0, prop_ratio_t0_m1 = (
            y[ind],
            t[ind],
            p_x[ind],
            prop_ratio_t1_m0[ind],
            prop_ratio_t0_m1[ind],
        )

        # importance weighting
        y1m1 = (y * t / p_x) / np.mean(t / p_x)
        y1m0 = (y * t * prop_ratio_t1_m0) / np.mean(t * prop_ratio_t1_m0)
        y0m0 = (y * (1 - t) / (1 - p_x)) / np.mean((1 - t) / (1 - p_x))
        y0m1 = (y * (1 - t) * prop_ratio_t0_m1) / np.mean((1 - t) * prop_ratio_t0_m1)

        return y0m0, y0m1, y1m0, y1m1
