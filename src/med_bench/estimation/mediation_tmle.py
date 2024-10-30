import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from med_bench.estimation.base import Estimator
from med_bench.nuisances.utils import _get_regressor
from med_bench.utils.decorators import fitted

ALPHA = 10


class TMLE(Estimator):
    """Implementation of targeted maximum likelihood estimator

    Parameters
    ----------
        settings (dict): dictionnary of parameters
        lbda (float): regularization parameter
        support_vec_tol (float): tolerance for discarding non-supporting vectors
            if |alpha_i| < support_vec_tol * lbda then vector is discarded
        verbose (int): in {0, 1}
    """

    def __init__(self, procedure, ratio, clip, normalized, **kwargs):
        super().__init__(**kwargs)

        self._crossfit = 0
        self._procedure = procedure
        self._ratio = ratio
        self._clip = clip
        self._normalized = normalized

    def _one_step_correction_direct(self, t, m, x, y):

        n = t.shape[0]
        t, m, x, y = self.resize(t, m, x, y)
        t0 = np.zeros((n))
        t1 = np.ones((n))

        # estimate mediator densities
        if self._ratio == 'density':
            f_m0x, f_m1x = self._estimate_mediator_probability(t, m, x, y)
            p_x = self._estimate_treatment_propensity_x(t, m, x)
            ratio = f_m0x / (p_x * f_m1x)

        elif self._ratio == 'propensities':
            p_x, p_xm = self._estimate_treatment_probabilities(t, m, x)
            ratio = (1-p_xm) / ((1 - p_x) * p_xm)

        h_corrector = t * ratio - (1 - t)/(1 - p_x)

        x_t_mr = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t, m]])
        mu_tmx = self._regressor_y.predict(x_t_mr)
        # import pdb; pdb.set_trace()
        reg = LinearRegression(fit_intercept=False).fit(
            h_corrector.reshape(-1, 1), (y-mu_tmx).squeeze())
        epsilon_h = reg.coef_
        # epsilon_h = 0
        print(epsilon_h)

        x_t0_m = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t0, m]])
        x_t1_m = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t1, m]])

        mu_t0_mx = self._regressor_y.predict(x_t0_m)
        h_corrector_t0 = t0 * ratio - (1 - t0)/(1 - p_x)
        mu_t1_mx = self._regressor_y.predict(x_t1_m)
        h_corrector_t1 = t1 * ratio - (1 - t1)/(1 - p_x)
        mu_t0_mx_star = mu_t0_mx + epsilon_h * h_corrector_t0
        mu_t1_mx_star = mu_t1_mx + epsilon_h * h_corrector_t1

        regressor_y = _get_regressor(self._regularize,
                                     self._use_forest)
        reg_cross = clone(regressor_y)
        reg_cross.fit(x[t == 0], (mu_t1_mx_star[t == 0] -
                      mu_t0_mx_star[t == 0]).squeeze())

        theta_0 = reg_cross.predict(x)
        c_corrector = (1 - t)/(1 - p_x)
        reg = LinearRegression(fit_intercept=False).fit(
            c_corrector.reshape(-1, 1)[t == 0], (mu_t1_mx_star[t == 0] - y[t == 0]-theta_0[t == 0]).squeeze())
        epsilon_c = reg.coef_

        theta_0_star = theta_0 + epsilon_c*c_corrector
        theta_0_star = np.mean(theta_0_star)

        return theta_0_star

    def _one_step_correction_indirect(self, t, m, x, y):

        n = t.shape[0]
        t, m, x, y = self.resize(t, m, x, y)
        t0 = np.zeros((n))
        t1 = np.ones((n))

        # estimate mediator densities
        if self._ratio == 'density':
            f_m0x, f_m1x = self._estimate_mediator_probability(t, m, x, y)
            p_x = self._estimate_treatment_propensity_x(t, m, x)
            ratio = f_m0x / (p_x * f_m1x)

        elif self._ratio == 'propensities':
            p_x, p_xm = self._estimate_treatment_probabilities(t, m, x)
            ratio = (1-p_xm) / ((1 - p_x) * p_xm)

        h_corrector = t / p_x - t * ratio

        x_t_mr = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t, m]])
        mu_tmx = self._regressor_y.predict(x_t_mr)
        reg = LinearRegression(fit_intercept=False).fit(
            h_corrector.reshape(-1, 1), (y-mu_tmx).squeeze())
        epsilon_h = reg.coef_
        # epsilon_h = 0
        print('indirect', epsilon_h)

        # mu_t0_mx = self._regressor_y.predict(_get_interactions(False, x, t0, m))
        # h_corrector_t0 = t0 * f_t0 / (p_x * f_t1) - (1 - t0)/(1 - p_x)

        x_t1_m = np.hstack(
            [var.reshape(-1, 1) if len(var.shape) == 1 else var for var in [x, t1, m]])

        mu_t1_mx = self._regressor_y.predict(x_t1_m)
        h_corrector_t1 = t1 / p_x - t1 * ratio
        # mu_t0_mx_star = mu_t0_mx + epsilon_h * h_corrector_t0
        mu_t1_mx_star = mu_t1_mx + epsilon_h * h_corrector_t1

        regressor_y = _get_regressor(self._regularize,
                                     self._use_forest)

        # reg_cross.fit(x, (mu_t1_mx_star).squeeze())

        # omega_t = reg_cross.predict(x)
        # c_corrector = (2*t - 1)/p_x[:, None]
        # reg = LinearRegression(fit_intercept=False).fit(c_corrector.reshape(-1, 1)[t==0], (mu_t1_mx_star - omega_t).squeeze())
        # epsilon_c = reg.coef_

        reg_cross = clone(regressor_y)
        reg_cross.fit(x[t == 0], mu_t1_mx_star[t == 0])
        omega_t0x = reg_cross.predict(x)
        c_corrector_t0 = (2*t0 - 1) / p_x[:, None]

        reg = LinearRegression(fit_intercept=False).fit(
            c_corrector_t0[t == 0], (mu_t1_mx_star[t == 0]-omega_t0x[t == 0]).squeeze())
        epsilon_c_t0 = reg.coef_
        # epsilon_c_t0 = 0
        omega_t0x_star = omega_t0x + epsilon_c_t0*c_corrector_t0

        reg_cross = clone(regressor_y)
        reg_cross.fit(x[t == 1], y[t == 1])
        omega_t1x = reg_cross.predict(x)
        c_corrector_t1 = (2*t1 - 1) / p_x[:, None]
        reg = LinearRegression(fit_intercept=False).fit(
            c_corrector_t1[t == 1], (y[t == 1]-omega_t1x[t == 1]).squeeze())
        epsilon_c_t1 = reg.coef_
        # epsilon_c_t1 = 0
        omega_t1x_star = omega_t1x + epsilon_c_t1*c_corrector_t1
        delta_1 = np.mean(omega_t1x_star - omega_t0x_star)

        return delta_1

    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

        """
        # bucketize if needed
        t, m, x, y = self._resize(t, m, x, y)

        self._fit_treatment_propensity_x_nuisance(t, x)
        self._fit_conditional_mean_outcome_nuisance(t, m, x, y)

        if self._ratio == 'density':
            self._fit_mediator_nuisance(t, m, x)

        elif self._ratio == 'propensities':
            self._fit_treatment_propensity_xm_nuisance(t, m, x)

        self._fitted = True

        if self.verbose:
            print("Nuisance models fitted")

        return self

    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        """
        theta_0 = self._one_step_correction_direct(t, m, x, y)
        delta_1 = self._one_step_correction_indirect(t, m, x, y)
        total_effect = theta_0 + delta_1
        direct_effect_treated = np.copy(theta_0)
        direct_effect_control = theta_0
        indirect_effect_treated = delta_1
        indirect_effect_control = np.copy(delta_1)

        causal_effects = {
            'total_effect': total_effect,
            'direct_effect_treated': direct_effect_treated,
            'direct_effect_control': direct_effect_control,
            'indirect_effect_treated': indirect_effect_treated,
            'indirect_effect_control': indirect_effect_control
        }
        return causal_effects
