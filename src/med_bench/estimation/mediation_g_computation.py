import numpy as np
import warnings

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import is_array_binary, is_array_integer


class GComputation(Estimator):
    """GComputation estimation method class"""

    def __init__(self, regressor, classifier, integration="implicit", **kwargs):
        """Initializes GComputation estimation method

        Parameters
        ----------
        regressor
            Regressor used for mu estimation, can be any object with a fit and
            predict method
        classifier
            Classifier used for propensity estimation, can be any object with a
            fit and predict_proba method
        integration
            str used to define which integration to perform for estimating the
            cross conditional mean outcome
        """
        super().__init__(**kwargs)

        assert hasattr(regressor, "fit"), "The model does not have a 'fit' method."
        assert hasattr(
            regressor, "predict"
        ), "The model does not have a 'predict' method."
        assert hasattr(classifier, "fit"), "The model does not have a 'fit' method."
        assert hasattr(
            classifier, "predict_proba"
        ), "The model does not have a 'predict_proba' method."
        self.regressor = regressor
        self.classifier = classifier
        self._integration = integration

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data"""
        t, m, x, y = self._resize(t, m, x, y)

        if self._integration == "explicit":
            self.discretizer.fit(m)
            m = self._discretize_mediators(m)
            self._fit_discrete_mediator_probability(t, m, x)
            self._fit_conditional_mean_outcome(t, m, x, y)

        elif self._integration == "implicit":
            self._fit_cross_conditional_mean_outcome(t, m, x, y)

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

        if self._integration == "explicit" and not is_array_integer(m):
            warnings.warn(
                "The explicit integration of the conditional mean outcome is strongly not advised for continuous mediators"
                "It is advised to set integration to 'implicit'.",
                UserWarning,
            )

        return self

    def _pointwise_estimate(self, t, m, x, y):
        """Estimates causal effect on data."""
        t, m, x, y = self._resize(t, m, x, y)

        if self._integration == "explicit":
            m = self._discretize_mediators(m) if not is_array_integer(m) else m

            f_0x, f_1x = self._estimate_discrete_mediator_probability_table(x)
            mu_0x, mu_1x = self._estimate_conditional_mean_outcome_table(x)
            y1m1, y0m0, y1m0, y0m1 = 0, 0, 0, 0

            for idx, m_anchor in enumerate(self.mediator_bins):

                f_0mx = f_0x[idx]
                f_1mx = f_1x[idx]
                mu_0mx = mu_0x[idx]
                mu_1mx = mu_1x[idx]

                y1m1 += mu_1mx * f_1mx
                y0m0 += mu_0mx * f_0mx
                y1m0 += mu_1mx * f_0mx
                y0m1 += mu_0mx * f_1mx

        elif self._integration == "implicit":
            _, _, y0m0, y0m1, y1m0, y1m1 = (
                self._estimate_cross_conditional_mean_outcome(m, x)
            )

        return y0m0, y0m1, y1m0, y1m1
