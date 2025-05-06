import numpy as np

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import is_array_integer
import warnings


class MultiplyRobust(Estimator):
    """Iniitializes Multiply Robust estimatation method class"""

    def __init__(
        self,
        regressor,
        classifier,
        clip: float,
        trim: float,
        prop_ratio="treatment",
        integration="implicit",
        normalized=True,
        **kwargs
    ):
        """Initializes MulitplyRobust estimatation method

        Parameters
        ----------
        regressor
            Regressor used for mu estimation, can be any object with a fit and predict method
        classifier
            Classifier used for propensity estimation, can be any object with a fit and predict_proba method
        clip : float
            Clipping value for propensity scores
        trim : float
            Trimming value for propensity scores
        prop_ratio : str
            prop_ratio to use for estimation, can be either 'mediator' or 'treatment'
        integration : str
            used to define which integration to perform for estimating the
            cross conditional mean outcome
        normalized : bool
            Whether to normalize the propensity scores
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

        self._clip = clip
        self._trim = trim
        assert prop_ratio in ["mediator", "treatment"]
        assert integration in ["implicit", "explicit"]
        self._prop_ratio = prop_ratio
        self._normalized = normalized
        self._integration = integration
        self.name = "MR"

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data"""
        t, m, x, y = self._resize(t, m, x, y)

        if self._prop_ratio == "mediator":
            self._fit_treatment_propensity_x(t, x)
            self._fit_mediator_probability(t, m, x)

        elif self._prop_ratio == "treatment":
            self._fit_treatment_propensity_x(t, x)
            self._fit_treatment_propensity_xm(t, m, x)

        if self._integration == "explicit":
            if self._prop_ratio == "treatment":
                self._fit_mediator_discretizer(m)
            m_label, m_discrete_value = self._discretize_mediators(m)
            self._fit_discrete_mediator_probability(t, m_label, x)
            self._fit_conditional_mean_outcome(t, m_discrete_value, x, y)

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
        """Point wise estimate of the causal effect on data"""

        # Format checking
        t, m, x, y = self._resize(t, m, x, y)

        p_x = self._estimate_treatment_propensity_x(x)
        p_x = np.clip(p_x, self._clip, 1 - self._clip)

        if self._prop_ratio == "mediator":
            f_m0x, f_m1x = self._estimate_mediator_probability(x, m)
            f_m0x = np.clip(f_m0x, self._clip, None)
            f_m1x = np.clip(f_m1x, self._clip, None)
            prop_ratio_t1_m0 = f_m0x / (p_x * f_m1x)
            prop_ratio_t0_m1 = f_m1x / ((1 - p_x) * f_m0x)

        elif self._prop_ratio == "treatment":
            p_xm = self._estimate_treatment_propensity_xm(m, x)
            p_xm = np.clip(p_xm, self._clip, 1 - self._clip)
            prop_ratio_t1_m0 = (1 - p_xm) / ((1 - p_x) * p_xm)
            prop_ratio_t0_m1 = p_xm / ((1 - p_xm) * p_x)

        if self._integration == "explicit":
            m_label, m_discrete_value = self._discretize_mediators(m)

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

            y1m1 = y1m1.mean()
            y0m0 = y0m0.mean()
            y1m0 = y1m0.mean()
            y0m1 = y0m1.mean()

            mu_0mx, mu_1mx = self._estimate_conditional_mean_outcome(
                x, m_discrete_value
            )

        elif self._integration == "implicit":
            mu_0mx, mu_1mx, y0m0, y0m1, y1m0, y1m1 = (
                self._estimate_cross_conditional_mean_outcome(m, x)
            )

        ind = (p_x > self._trim) & (p_x < (1 - self._trim))
        y, t, p_x, prop_ratio_t1_m0, prop_ratio_t0_m1 = (
            y[ind],
            t[ind],
            p_x[ind],
            prop_ratio_t1_m0[ind],
            prop_ratio_t0_m1[ind],
        )
        mu_0mx, mu_1mx = (
            mu_0mx[ind],
            mu_1mx[ind],
        )

        # score computing
        if self._normalized:
            sum_score_m1 = np.mean(t / p_x)
            sum_score_m0 = np.mean((1 - t) / (1 - p_x))
            sum_score_t1m0 = np.mean(t * prop_ratio_t1_m0)
            sum_score_t0m1 = np.mean((1 - t) * prop_ratio_t0_m1)

            y1m1 = (t / p_x * (y - y1m1)) / sum_score_m1 + y1m1
            y0m0 = ((1 - t) / (1 - p_x) * (y - y0m0)) / sum_score_m0 + y0m0

            y1m0 = (
                (t * prop_ratio_t1_m0 * (y - mu_1mx)) / sum_score_t1m0
                + ((1 - t) / (1 - p_x) * (mu_1mx - y1m0)) / sum_score_m0
                + y1m0
            )

            y0m1 = (
                ((1 - t) * prop_ratio_t0_m1 * (y - mu_0mx)) / sum_score_t0m1
                + t / p_x * (mu_0mx - y0m1) / sum_score_m1
                + y0m1
            )
        else:
            y1m1 = t / p_x * (y - y1m1) + y1m1
            y0m0 = (1 - t) / (1 - p_x) * (y - y0m0) + y0m0

            y1m0 = (
                t * prop_ratio_t1_m0 * (y - mu_1mx)
                + (1 - t) / (1 - p_x) * (mu_1mx - y1m0)
                + y1m0
            )
            y0m1 = (
                (1 - t) * prop_ratio_t0_m1 * (y - mu_0mx)
                + t / p_x * (mu_0mx - y0m1)
                + y0m1
            )
        return y0m0, y0m1, y1m0, y1m1
