import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from med_bench.estimation.base import Estimator
from med_bench.utils.decorators import fitted
from med_bench.utils.utils import is_array_binary

ALPHA = 10


class TMLE(Estimator):
    """Implementation of targeted maximum likelihood estimation method class"""

    def __init__(self, regressor, classifier, prop_ratio, **kwargs):
        """_summary_

        Parameters
        ----------
        regressor
            Regressor used for mu estimation, can be any object with a fit and predict method
        classifier
            Classifier used for propensity estimation, can be any object with a fit and predict_proba method
        prop_ratio : str
            prop_ratio to use for estimation, can be either 'mediator' or 'treatment'
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

        assert prop_ratio in ["mediator", "treatment"]
        self._prop_ratio = prop_ratio

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data"""
        # bucketize if needed
        t, m, x, y = self._resize(t, m, x, y)

        if (not is_array_binary(m)) and (self._prop_ratio == "mediator"):
            raise ValueError(
                "The option mediator 'mediator' in TMLE is supported only for 1D binary mediator"
            )

        self._fit_treatment_propensity_x(t, x)
        self._fit_conditional_mean_outcome(t, m, x, y)

        if self._prop_ratio == "mediator":
            self._fit_mediator_probability(t, m, x)

        elif self._prop_ratio == "treatment":
            self._fit_treatment_propensity_xm(t, m, x)

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

        return self
        
    def _one_step_correction_direct(self, t, m, x, y):
        """Implements the one step correction for the estimation of the natural
        direct effect with the prop_ratio of mediator densities or treatment
        propensities.

        """
        n = t.shape[0]
        t, m, x, y = self._resize(t, m, x, y)
        t0 = np.zeros((n))
        t1 = np.ones((n))

        # estimate mediator densities
        if self._prop_ratio == "mediator":
            f_m0x, f_m1x = self._estimate_mediator_probability(x, m)
            p_x = self._estimate_treatment_propensity_x(x)
            prop_ratio = f_m0x / (p_x * f_m1x)

        elif self._prop_ratio == "treatment":
            p_x = self._estimate_treatment_propensity_x(x)
            p_xm = self._estimate_treatment_propensity_xm(m, x)
            prop_ratio = (1 - p_xm) / ((1 - p_x) * p_xm)

        # estimation of corrective features for the conditional mean outcome
        h_corrector = t * prop_ratio - (1 - t) / (1 - p_x)

        x_t_mr = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t, m]]
        )
        mu_tmx = self._regressor_y.predict(x_t_mr)

        # regress with OLS the error of conditional mean outcome regressor on
        # corrective features
        reg = LinearRegression(fit_intercept=False).fit(
            h_corrector.reshape(-1, 1), (y - mu_tmx).squeeze()
        )
        # corrective coefficient epsilon
        epsilon_h = reg.coef_

        x_t0_m = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t0, m]]
        )
        x_t1_m = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t1, m]]
        )

        # one step corrected conditional mean outcomes
        mu_t0_mx = self._regressor_y.predict(x_t0_m)
        h_corrector_t0 = t0 * prop_ratio - (1 - t0) / (1 - p_x)
        mu_t1_mx = self._regressor_y.predict(x_t1_m)
        h_corrector_t1 = t1 * prop_ratio - (1 - t1) / (1 - p_x)
        mu_t0_mx_star = mu_t0_mx + epsilon_h * h_corrector_t0
        mu_t1_mx_star = mu_t1_mx + epsilon_h * h_corrector_t1

        # estimation of natural direct effect
        reg_cross = clone(self.regressor)
        reg_cross.fit(
            x[t == 0], (mu_t1_mx_star[t == 0] - mu_t0_mx_star[t == 0]).squeeze()
        )

        theta_0 = reg_cross.predict(x)

        # one step correction of the natural direct effect
        c_corrector = (1 - t) / (1 - p_x)
        reg = LinearRegression(fit_intercept=False).fit(
            c_corrector.reshape(-1, 1)[t == 0],
            (mu_t1_mx_star[t == 0] - y[t == 0] - theta_0[t == 0]).squeeze(),
        )
        epsilon_c = reg.coef_

        theta_0_star = theta_0 + epsilon_c * c_corrector
        theta_0_star = np.mean(theta_0_star)

        return theta_0_star

    def _one_step_correction_indirect(self, t, m, x, y):
        """Implements the one step correction for the estimation of the natural
        indirect effect with the prop_ratio of mediator densities or treatment
        propensities.

        """
        n = t.shape[0]
        t, m, x, y = self._resize(t, m, x, y)
        t0 = np.zeros((n))
        t1 = np.ones((n))

        # estimate mediator densities
        if self._prop_ratio == "mediator":
            f_m0x, f_m1x = self._estimate_mediator_probability(x, m)
            p_x = self._estimate_treatment_propensity_x(x)
            prop_ratio = f_m0x / (p_x * f_m1x)

        elif self._prop_ratio == "treatment":
            p_x = self._estimate_treatment_propensity_x(x)
            p_xm = self._estimate_treatment_propensity_xm(m, x)
            prop_ratio = (1 - p_xm) / ((1 - p_x) * p_xm)

        # estimation of corrective features for the conditional mean outcome
        h_corrector = t / p_x - t * prop_ratio

        x_t_mr = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t, m]]
        )
        mu_tmx = self._regressor_y.predict(x_t_mr)

        # regress with OLS the error of conditional mean outcome regressor on
        # corrective features
        reg = LinearRegression(fit_intercept=False).fit(
            h_corrector.reshape(-1, 1), (y - mu_tmx).squeeze()
        )

        # corrective coefficient epsilon
        epsilon_h = reg.coef_

        x_t1_m = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t1, m]]
        )

        # one step corrected conditional mean outcomes
        mu_t1_mx = self._regressor_y.predict(x_t1_m)
        h_corrector_t1 = t1 / p_x - t1 * prop_ratio
        mu_t1_mx_star = mu_t1_mx + epsilon_h * h_corrector_t1

        # cross conditional mean outcome control
        reg_cross = clone(self.regressor)
        reg_cross.fit(x[t == 0], mu_t1_mx_star[t == 0])
        omega_t0x = reg_cross.predict(x)

        # one step corrected cross conditional mean outcome for control
        c_corrector_t0 = (2 * t0 - 1) / p_x[:, None]
        reg = LinearRegression(fit_intercept=False).fit(
            c_corrector_t0[t == 0],
            (mu_t1_mx_star[t == 0] - omega_t0x[t == 0]).squeeze(),
        )
        epsilon_c_t0 = reg.coef_
        omega_t0x_star = omega_t0x + epsilon_c_t0 * c_corrector_t0

        # cross conditional mean outcome treated
        reg_cross = clone(self.regressor)
        reg_cross.fit(x[t == 1], y[t == 1])
        omega_t1x = reg_cross.predict(x)

        # one step corrected cross conditional mean outcome for treated
        c_corrector_t1 = (2 * t1 - 1) / p_x[:, None]
        reg = LinearRegression(fit_intercept=False).fit(
            c_corrector_t1[t == 1], (y[t == 1] - omega_t1x[t == 1]).squeeze()
        )
        epsilon_c_t1 = reg.coef_
        omega_t1x_star = omega_t1x + epsilon_c_t1 * c_corrector_t1

        # natural indirect effect
        delta_1 = np.mean(omega_t1x_star - omega_t0x_star)

        return delta_1


    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data"""
        theta_0 = self._one_step_correction_direct(t, m, x, y)
        delta_1 = self._one_step_correction_indirect(t, m, x, y)
        total_effect = theta_0 + delta_1
        direct_effect_treated = None
        direct_effect_control = theta_0
        indirect_effect_treated = delta_1
        indirect_effect_control = None

        causal_effects = {
            "total_effect": total_effect,
            "direct_effect_treated": direct_effect_treated,
            "direct_effect_control": direct_effect_control,
            "indirect_effect_treated": indirect_effect_treated,
            "indirect_effect_control": indirect_effect_control,
        }
        return causal_effects
