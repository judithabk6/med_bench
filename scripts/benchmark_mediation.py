import time
from src.benchmark_mediation import *
from rpy2.rinterface import RRuntimeError


def simulate_data(n, rg, setting, dim_x_observed=1,
                  dim_m_observed=1,
                  seed=None, dim_x=1, dim_m=1, type_m='binary', sigma_y=0.5,
                  sigma_m=0.5, beta_t_factor=1, strict=False):
    """
    """
    rg_coef = default_rng(seed)
    x_observed = rg.standard_normal(n * dim_x_observed)\
                   .reshape((n, dim_x_observed))
    alphas = np.zeros(dim_x_observed)
    alphas[:dim_x] = 1 / dim_x
    p_t = expit(alphas.dot(x_observed.T))
    t = rg.binomial(1, p_t, n).reshape(-1, 1)
    t0 = np.zeros((n, 1))
    t1 = np.ones((n, 1))
    if (setting == 'linear'):
        if (type_m == 'binary'):
            beta_x = np.zeros((dim_x_observed, 1))
            beta_x[:dim_x] = 1 / dim_x
            beta_t = 2 * beta_t_factor
            p_m0 = expit(x_observed.dot(beta_x) + beta_t * t0).flatten()
            p_m1 = expit(x_observed.dot(beta_x) + beta_t * t1).flatten()
            pre_m = rg.random(n)
            m0 = ((pre_m < p_m0) * 1).reshape(-1, 1)
            m1 = ((pre_m < p_m1) * 1).reshape(-1, 1)
            m_2d = np.hstack((m0, m1))
            m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
            gamma_m = np.array([[1]])
        if (type_m == 'continuous'):
            beta_x = np.zeros((dim_x_observed, dim_m_observed))
            for i in range(dim_m_observed):
                idx = rg_coef.choice(dim_x, min(2, dim_x), replace=False)
                beta_x[idx, i] = 1
            beta_t = (np.array([[i%3 for i in range(0, dim_m_observed)]]) + 1) * beta_t_factor
            if strict:
                beta_t[0,dim_m:] = 0
            m = x_observed.dot(beta_x) + t.dot(beta_t) + sigma_m * rg.standard_normal((n, dim_m_observed))
            gamma_m = np.zeros((dim_m_observed, 1))
            gamma_m[0:dim_m] = 0.5 / dim_m
        gamma_x = np.zeros((dim_x_observed, 1))
        gamma_x[:dim_x] = (np.arange(1, dim_x+1) / dim_x**2).reshape(-1, 1)
        gamma_t = 1.2
        y = x_observed.dot(gamma_x) + gamma_t * t + m.dot(gamma_m) + sigma_y * rg.standard_normal((n, 1))

    elif (setting == "mis_specification_M"):
        if type_m == 'binary':
            beta_x = np.zeros((dim_x_observed, 1))
            beta_x[:dim_x] = 1 / dim_x
            beta_t = 2 * beta_t_factor
            p_m0 = expit(1/(1 + np.exp(-x_observed.dot(beta_x))) + beta_t * t0).flatten()
            p_m1 = expit(1/(1 + np.exp(-x_observed.dot(beta_x))) + beta_t * t1).flatten()
            pre_m = rg.random(n)
            m0 = ((pre_m < p_m0) * 1).reshape(-1, 1)
            m1 = ((pre_m < p_m1) * 1).reshape(-1, 1)
            m_2d = np.hstack((m0, m1))
            m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
            gamma_m = np.array([[1]])
        else:
            beta_x = np.zeros((dim_x_observed, dim_m_observed))
            for i in range(dim_m_observed):
                idx = rg_coef.choice(dim_x, min(2, dim_x), replace=False)
                beta_x[idx, i] = 0.25
            beta_t = (np.array([[i%3 for i in range(0, dim_m_observed)]]) + 1) * beta_t_factor
            if strict:
                beta_t[0,dim_m:] = 0
            m = 1/(1+np.exp(-x_observed.dot(beta_x))) + t.dot(beta_t) + sigma_m * rg.standard_normal((n, dim_m_observed))
            gamma_m = np.zeros((dim_m_observed, 1))
            gamma_m[0:dim_m] = 0.5 / dim_m
        gamma_x = np.zeros((dim_x_observed, 1))
        gamma_x[:dim_x] = (np.arange(1, dim_x+1) / dim_x**2).reshape(-1, 1)
        gamma_t = 1.2
        y = x_observed.dot(gamma_x) + gamma_t * t + m.dot(gamma_m) + sigma_y * rg.standard_normal((n, 1))

    elif (setting == "severe_mis_specification_M"):
        if type_m == 'binary':
            beta_x = np.zeros((dim_x_observed, 1))
            beta_x[:dim_x] = 1 / dim_x
            beta_t = 2 * beta_t_factor
            p_m0 = expit(3*np.sin(3*x_observed.dot(beta_x)) + beta_t * t0).flatten()
            p_m1 = expit(3*np.sin(3*x_observed.dot(beta_x)) + beta_t * t1).flatten()
            pre_m = rg.random(n)
            m0 = ((pre_m < p_m0) * 1).reshape(-1, 1)
            m1 = ((pre_m < p_m1) * 1).reshape(-1, 1)
            m_2d = np.hstack((m0, m1))
            m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
            gamma_m = np.array([[1]])
        else:
            beta_x = np.zeros((dim_x_observed, dim_m_observed))
            for i in range(dim_m_observed):
                idx = rg_coef.choice(dim_x, min(2, dim_x), replace=False)
                beta_x[idx, i] = 0.25
            beta_t = (np.array([[i%3 for i in range(0, dim_m_observed)]]) + 1) * beta_t_factor
            if strict:
                beta_t[0,dim_m:] = 0
            m = 3*np.sin(3*x_observed.dot(beta_x)) + t.dot(beta_t) + sigma_m * rg.standard_normal((n, dim_m_observed))
            gamma_m = np.zeros((dim_m_observed, 1))
            gamma_m[0:dim_m] = 0.5 / dim_m
        gamma_x = np.zeros((dim_x_observed, 1))
        gamma_x[:dim_x] = (np.arange(1, dim_x+1) / dim_x**2).reshape(-1, 1)
        gamma_t = 1.2
        y = x_observed.dot(gamma_x) + gamma_t * t + m.dot(gamma_m) + sigma_y * rg.standard_normal((n, 1))

    elif (setting == "mis_specification_Y"):
        if (type_m == 'binary'):
            beta_x = np.zeros((dim_x_observed, 1))
            beta_x[:dim_x] = 1 / dim_x
            beta_t = 2 * beta_t_factor
            p_m0 = expit(x_observed.dot(beta_x) + beta_t * t0).flatten()
            p_m1 = expit(x_observed.dot(beta_x) + beta_t * t1).flatten()
            pre_m = rg.random(n)
            m0 = ((pre_m < p_m0) * 1).reshape(-1, 1)
            m1 = ((pre_m < p_m1) * 1).reshape(-1, 1)
            m_2d = np.hstack((m0, m1))
            m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
            gamma_m = np.array([[1]])
        if (type_m == 'continuous'):
            beta_x = np.zeros((dim_x_observed, dim_m_observed))
            for i in range(dim_m_observed):
                idx = rg_coef.choice(dim_x, min(2, dim_x), replace=False)
                beta_x[idx, i] = 0.25
            beta_t = (np.array([[i%3 for i in range(0, dim_m_observed)]]) + 1) * beta_t_factor
            if strict:
                beta_t[0,dim_m:] = 0
            m = x_observed.dot(beta_x) + t.dot(beta_t) + sigma_m * rg.standard_normal((n, dim_m_observed))
            gamma_m = np.zeros((dim_m_observed, 1))
            gamma_m[0:dim_m] = 0.5 / dim_m
        gamma_x = np.zeros((dim_x_observed, 1))
        gamma_x[:dim_x] = (np.arange(1, dim_x+1) / dim_x**2).reshape(-1, 1)
        gamma_t = 1.2
        y = 1/(1+np.exp(-x_observed.dot(gamma_x))) + gamma_t * t + m.dot(gamma_m) + sigma_y * rg.standard_normal((n, 1))

    elif (setting == "severe_mis_specification_Y"):
        if (type_m == 'binary'):
            beta_x = np.zeros((dim_x_observed, 1))
            beta_x[:dim_x] = 1 / dim_x
            beta_t = 2 * beta_t_factor
            p_m0 = expit(x_observed.dot(beta_x) + beta_t * t0).flatten()
            p_m1 = expit(x_observed.dot(beta_x) + beta_t * t1).flatten()
            pre_m = rg.random(n)
            m0 = ((pre_m < p_m0) * 1).reshape(-1, 1)
            m1 = ((pre_m < p_m1) * 1).reshape(-1, 1)
            m_2d = np.hstack((m0, m1))
            m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
            gamma_m = np.array([[1]])
        if (type_m == 'continuous'):
            beta_x = np.zeros((dim_x_observed, dim_m_observed))
            for i in range(dim_m_observed):
                idx = rg_coef.choice(dim_x, min(2, dim_x), replace=False)
                beta_x[idx, i] = 0.25
            beta_t = (np.array([[i%3 for i in range(0, dim_m_observed)]]) + 1) * beta_t_factor
            if strict:
                beta_t[0,dim_m:] = 0
            m = x_observed.dot(beta_x) + t.dot(beta_t) + sigma_m * rg.standard_normal((n, dim_m_observed))
            gamma_m = np.zeros((dim_m_observed, 1))
            gamma_m[0:dim_m] = 0.5 / dim_m
        gamma_x = np.zeros((dim_x_observed, 1))
        gamma_x[:dim_x] = (np.arange(1, dim_x+1) / dim_x**2).reshape(-1, 1)
        gamma_t = 1.2
        y = 3*np.sin(3*x_observed.dot(gamma_x)) + gamma_t * t + m.dot(gamma_m) + sigma_y * rg.standard_normal((n, 1))

    elif (setting == 'mis_specification_MY'):
        if type_m == 'binary':
            beta_x = np.zeros((dim_x_observed, 1))
            beta_x[:dim_x] = 0.5 / dim_x
            beta_t = 2 * beta_t_factor
            p_m0 = expit(1/(1 + np.exp(-x_observed.dot(beta_x))) + beta_t * t0).flatten()
            p_m1 = expit(1/(1 + np.exp(-x_observed.dot(beta_x))) + beta_t * t1).flatten()
            pre_m = rg.random(n)
            m0 = ((pre_m < p_m0) * 1).reshape(-1, 1)
            m1 = ((pre_m < p_m1) * 1).reshape(-1, 1)
            m_2d = np.hstack((m0, m1))
            m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
            gamma_m = np.array([[1]])
        else:
            beta_x = np.zeros((dim_x_observed, dim_m_observed))
            for i in range(dim_m_observed):
                idx = rg_coef.choice(dim_x, min(2, dim_x), replace=False)
                beta_x[idx, i] = 0.25
            beta_t = (np.array([[i%3 for i in range(0, dim_m_observed)]]) + 1) * beta_t_factor
            if strict:
                beta_t[0,dim_m:] = 0
            m = 1/(1+np.exp(-x_observed.dot(beta_x))) + t.dot(beta_t) + sigma_m * rg.standard_normal((n, dim_m_observed))
            gamma_m = np.zeros((dim_m_observed, 1))
            gamma_m[0:dim_m] = 0.5 / dim_m
        gamma_x = np.zeros((dim_x_observed, 1))
        gamma_x[:dim_x] = (np.arange(1, dim_x+1) / dim_x**2).reshape(-1, 1)
        gamma_t = 1.2
        y = 1/(1+np.exp(-x_observed.dot(gamma_x))) + gamma_t * t + m.dot(gamma_m) + sigma_y * rg.standard_normal((n, 1))

    elif (setting == 'severe_mis_specification_MY'):
        if type_m == 'binary':
            beta_x = np.zeros((dim_x_observed, 1))
            beta_x[:dim_x] = 0.5 / dim_x
            beta_t = 2 * beta_t_factor
            p_m0 = expit(3*np.sin(3*x_observed.dot(beta_x)) + beta_t * t0).flatten()
            p_m1 = expit(3*np.sin(3*x_observed.dot(beta_x)) + beta_t * t1).flatten()
            pre_m = rg.random(n)
            m0 = ((pre_m < p_m0) * 1).reshape(-1, 1)
            m1 = ((pre_m < p_m1) * 1).reshape(-1, 1)
            m_2d = np.hstack((m0, m1))
            m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
            gamma_m = np.array([[1]])
        else:
            beta_x = np.zeros((dim_x_observed, dim_m_observed))
            for i in range(dim_m_observed):
                idx = rg_coef.choice(dim_x, min(2, dim_x), replace=False)
                beta_x[idx, i] = 0.25
            beta_t = (np.array([[i%3 for i in range(0, dim_m_observed)]]) + 1) * beta_t_factor
            if strict:
                beta_t[0,dim_m:] = 0
            m = 3*np.sin(3*x_observed.dot(beta_x)) + t.dot(beta_t) + sigma_m * rg.standard_normal((n, dim_m_observed))
            gamma_m = np.zeros((dim_m_observed, 1))
            gamma_m[0:dim_m] = 0.5 / dim_m
        gamma_x = np.zeros((dim_x_observed, 1))
        gamma_x[:dim_x] = (np.arange(1, dim_x+1) / dim_x**2).reshape(-1, 1)
        gamma_t = 1.2
        y = 3*np.sin(3*x_observed.dot(gamma_x)) + gamma_t * t + m.dot(gamma_m) + sigma_y * rg.standard_normal((n, 1))

    if type_m == 'continuous':
        true_indirect_effect = np.sum(beta_t * gamma_m.flatten())
    elif (type_m == 'binary') and (dim_m_observed==1):
        true_indirect_effect = np.mean((p_m1 - p_m0) * gamma_m.flatten())
    else:
        raise NotImplementedError
    true_direct_effect = gamma_t
    true_total_effect = true_direct_effect + true_indirect_effect
    return x_observed, t, m, y, (true_total_effect, true_direct_effect, true_indirect_effect)


def get_estimation(x, t, m, y, estimator, config):
    if estimator == 'huber_IPW_R':
        x_r, t_r, m_r, y_r = [_convert_array_to_R(uu) for uu in (x, t, m, y)]
        output_w = causalweight.medweight(y=y_r, d=t_r, m=m_r, x=x_r, trim=0.0,
                                          ATET="FALSE", logit="TRUE", boot=2)
        raw_res_R = np.array(output_w.rx2('results'))
        effects = raw_res_R[0, :]
    if estimator == 'coefficient_product':
        effects = ols_mediation(y, t, m, x)
    if estimator == 'huber_ipw_noreg':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=False, forest=False, crossfit=0,
                            clip=0.0, calibration=False)
    if estimator == 'huber_ipw_noreg_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=False, forest=False, crossfit=2,
                            clip=0.0, calibration=False)
    if estimator == 'huber_ipw_reg':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=0,
                            clip=0.01, calibration=False)
    if estimator == 'huber_ipw_reg_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=2,
                            clip=0.01, calibration=False)
    if estimator == 'huber_ipw_reg_calibration':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=0,
                            clip=0.01, calibration=True)
    if estimator == 'huber_ipw_reg_calibration_iso':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=0,
                            clip=0.01, calibration=True,
                            calib_method='isotonic')
    if estimator == 'huber_ipw_reg_calibration_iso_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=2,
                            clip=0.01, calibration=True)
    if estimator == 'huber_ipw_reg_calibration_iso_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=False, crossfit=2,
                            clip=0.01, calibration=True,
                            calib_method='isotonic')
    if estimator == 'huber_ipw_forest':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=0,
                            clip=0.01, calibration=False)
    if estimator == 'huber_ipw_forest_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=2,
                            clip=0.01, calibration=False)
    if estimator == 'huber_ipw_forest_calibration':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=0,
                            clip=0.01, calibration=True)
    if estimator == 'huber_ipw_forest_calibration_iso':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=0,
                            clip=0.01, calibration=True,
                            calib_method='isotonic')
    if estimator == 'huber_ipw_forest_calibration_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=2,
                            clip=0.01, calibration=True)
    if estimator == 'huber_ipw_forest_calibration_iso_cf':
        effects = huber_IPW(y, t, m, x, None, None, trim=0, logit=True,
                            regularization=True, forest=True, crossfit=2,
                            clip=0.01, calibration=True,
                            calib_method='isotonic')
    if estimator == 'g_computation_noreg':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False, 
                                    forest=False, crossfit=0,
                                    regularization=False, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_noreg_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=2,
                                    regularization=False, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=0,
                                    regularization=True, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=2,
                                    regularization=True, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_calibration':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=0,
                                    regularization=True, calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_calibration_iso':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=0,
                                    regularization=True, calibration=True,
                                    calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_calibration_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=2,
                                    regularization=True, calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_reg_calibration_iso_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=False, crossfit=2,
                                    regularization=True, calibration=True,
                                    calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=0,
                                    regularization=True, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=2,
                                    regularization=True, calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_calibration':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=0,
                                    regularization=True, calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_calibration_iso':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=0,
                                    regularization=True, calibration=True,
                                    calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_calibration_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=2,
                                    regularization=True, calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'g_computation_forest_calibration_iso_cf':
        if config in (0, 1, 2):
            effects = g_computation(y, t, m, x, interaction=False,
                                    forest=True, crossfit=2,
                                    regularization=True, calibration=True,
                                    calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_noreg':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=0, clip=0.0,
                                                regularization=False,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_noreg_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=2, clip=0.0,
                                                regularization=False,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_calibration':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_calibration_sio':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=True,
                                                calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_calibration_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_reg_calibration_iso_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=False,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=True,
                                                calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_calibration':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_calibration_iso':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=0, clip=0.01,
                                                regularization=True,
                                                calibration=True,
                                                calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_calibration_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=True)
        else:
            effects = [np.nan] * 5
    if estimator == 'multiply_robust_forest_calibration_iso_cf':
        if config in (0, 1, 2):
            effects = multiply_robust_efficient(y, t, m.astype(int), x, interaction=False,
                                                forest=True,
                                                crossfit=2, clip=0.01,
                                                regularization=True,
                                                calibration=True,
                                                calib_method='isotonic')
        else:
            effects = [np.nan] * 5
    if estimator == 'simulation_based':
        if config in (0, 1, 2):
            effects = r_mediate(y, t, m, x, interaction=False)
        else:
            effects = [np.nan] * 5
    if estimator == 'DML_huber':
        if config > 0:
            effects = medDML(y, t, m, x, trim=0.0, order=1)
        else:
            effects = [np.nan] * 5
    if estimator == 'G_estimator':
        if config in (0, 1, 2):
            effects = g_estimator(y, t, m, x)
        else:
            effects = [np.nan] * 5
    return effects


dim_x_observed_list = [1, 5, 20, 5, 20, 20, 20]
dim_x_list = [1, 5, 5, 5, 5, 5, 5]
dim_m_observed_list = [1, 1, 1, 5, 5, 20, 20]
dim_m_list = [1, 1, 1, 5, 5, 5, 5]
strict_list = [False, False, False, False, False, False, True]
type_m_list = ['binary', 'binary', 'binary',
               'continuous', 'continuous', 'continuous', 'continuous']
setting_list = ['linear', 'mis_specification_M',
                'mis_specification_Y', 'mis_specification_MY',
                'severe_mis_specification_M',
                'severe_mis_specification_Y', 'severe_mis_specification_MY']

estimator_list = ['coefficient_product', 'huber_ipw_noreg', 'huber_ipw_reg',
                  'huber_ipw_reg_calibration', 'huber_ipw_forest',
                  'huber_ipw_forest_calibration',
                  'g_computation_noreg', 'g_computation_reg',
                  'g_computation_reg_calibration', 'g_computation_forest',
                  'g_computation_forest_calibration',
                  'multiply_robust_noreg', 'multiply_robust_reg',
                  'multiply_robust_reg_calibration', 'multiply_robust_forest',
                  'multiply_robust_forest_calibration',
                  'simulation_based', 'DML_huber', 'G_estimator',
                  'huber_ipw_noreg_cf', 'huber_ipw_reg_cf',
                  'huber_ipw_reg_calibration_cf', 'huber_ipw_forest_cf',
                  'huber_ipw_forest_calibration_cf',
                  'g_computation_noreg_cf', 'g_computation_reg_cf',
                  'g_computation_reg_calibration_cf', 'g_computation_forest_cf',
                  'g_computation_forest_calibration_cf',
                  'multiply_robust_noreg_cf', 'multiply_robust_reg_cf',
                  'multiply_robust_reg_calibration_cf', 'multiply_robust_forest_cf',
                  'multiply_robust_forest_calibration_cf']
rep_nb = 30
res_list = list()
for ii in range(rep_nb):
    print("iteration", ii)
    for config in range(7):
        for setting in setting_list:
            for n in (500, 1000, 10000):
                for beta_t_factor in (0.1, 1, 5):
                    rg = default_rng(config * (setting_list.index(setting)+1) + n * (ii+1) * int(beta_t_factor+1))
                    x, t, m, y, truth = simulate_data(
                        n, rg, setting,
                        dim_x_observed=dim_x_observed_list[config],
                        dim_m_observed=dim_m_observed_list[config],
                        seed=123 * (ii + 1),
                        dim_x=dim_x_list[config],
                        dim_m=dim_m_list[config],
                        type_m=type_m_list[config],
                        sigma_y=0.5, sigma_m=0.5, beta_t_factor=beta_t_factor,
                        strict=strict_list[config])
                    for estimator in estimator_list:
                        val_list = [config, setting, n, beta_t_factor]
                        val_list += list(truth)
                        val_list.append(estimator)
                        start = time.time()
                        try:
                            effects = get_estimation(x, t.ravel(), m, y.ravel(), estimator, config)
                        except RRuntimeError:
                            effects = [np.nan*6]
                        duration = time.time() - start
                        val_list += list(effects)
                        val_list.append(duration)
                        res_list.append(val_list)
    columns = ['configuration', 'setting', 'n', 'beta_t_factor', 'true_total_effect',
               'true_direct_effect', 'true_indirect_effect', 'estimator',
               'total_effect', 'direct_treated_effect', 'direct_control_effect',
               'indirect_treated_effect', 'indirect_control_effect', 'n_non_trimmed', 'duration']
    res_df = pd.DataFrame(res_list, columns=columns)
    res_df.to_csv('results/simulations/20211110_simulations.csv', index=False, sep='\t')


rep_nb = 30
res_list = list()
for ii in range(rep_nb):
    print("iteration", ii)
    for config in [6]:
        for setting in setting_list:
            for n in (500, 1000, 10000):
                for beta_t_factor in (0.1, 1, 5):
                    rg = default_rng(config * (setting_list.index(setting)+1) + n * (ii+1) * int(beta_t_factor+1))
                    x, t, m, y, truth = simulate_data(
                        n, rg, setting,
                        dim_x_observed=dim_x_observed_list[config],
                        dim_m_observed=dim_m_observed_list[config],
                        seed=123 * (ii + 1),
                        dim_x=dim_x_list[config],
                        dim_m=dim_m_list[config],
                        type_m=type_m_list[config],
                        sigma_y=0.5, sigma_m=0.5, beta_t_factor=beta_t_factor,
                        strict=strict_list[config])
                    for estimator in estimator_list:
                        val_list = [config, setting, n, beta_t_factor]
                        val_list += list(truth)
                        val_list.append(estimator)
                        start = time.time()
                        try:
                            effects = get_estimation(x, t.ravel(), m, y.ravel(), estimator, config)
                        except RRuntimeError:
                            effects = [np.nan*6]
                        duration = time.time() - start
                        val_list += list(effects)
                        val_list.append(duration)
                        res_list.append(val_list)
    columns = ['configuration', 'setting', 'n', 'beta_t_factor', 'true_total_effect',
               'true_direct_effect', 'true_indirect_effect', 'estimator',
               'total_effect', 'direct_treated_effect', 'direct_control_effect',
               'indirect_treated_effect', 'indirect_control_effect', 'n_non_trimmed', 'duration']
    res_df = pd.DataFrame(res_list, columns=columns)
    res_df.to_csv('results/simulations/20211123_simulations.csv', index=False, sep='\t')

estimator_list = ['huber_ipw_reg_calibration_iso',
                  'huber_ipw_forest_calibration_iso',
                  'g_computation_reg_calibration_iso',
                  'g_computation_forest_calibration_iso',
                  'multiply_robust_reg_calibration_iso',
                  'multiply_robust_forest_calibration_iso',
                  'huber_ipw_reg_calibration_iso_cf',
                  'huber_ipw_forest_calibration_iso_cf',
                  'g_computation_reg_calibration_iso_cf',
                  'g_computation_forest_calibration_iso_cf',
                  'multiply_robust_reg_calibration_iso_cf',
                  'multiply_robust_forest_calibration_iso_cf']
rep_nb = 30
res_list = list()
for ii in range(rep_nb):
    print("iteration", ii)
    for config in range(7):
        for setting in setting_list:
            for n in (500, 1000, 10000):
                for beta_t_factor in (0.1, 1, 5):
                    rg = default_rng(config * (setting_list.index(setting)+1) + n * (ii+1) * int(beta_t_factor+1))
                    x, t, m, y, truth = simulate_data(
                        n, rg, setting,
                        dim_x_observed=dim_x_observed_list[config],
                        dim_m_observed=dim_m_observed_list[config],
                        seed=123 * (ii + 1),
                        dim_x=dim_x_list[config],
                        dim_m=dim_m_list[config],
                        type_m=type_m_list[config],
                        sigma_y=0.5, sigma_m=0.5, beta_t_factor=beta_t_factor,
                        strict=strict_list[config])
                    for estimator in estimator_list:
                        val_list = [config, setting, n, beta_t_factor]
                        val_list += list(truth)
                        val_list.append(estimator)
                        start = time.time()
                        try:
                            effects = get_estimation(x, t.ravel(), m, y.ravel(), estimator, config)
                        except RRuntimeError:
                            effects = [np.nan*6]
                        duration = time.time() - start
                        val_list += list(effects)
                        val_list.append(duration)
                        res_list.append(val_list)
    columns = ['configuration', 'setting', 'n', 'beta_t_factor', 'true_total_effect',
               'true_direct_effect', 'true_indirect_effect', 'estimator',
               'total_effect', 'direct_treated_effect', 'direct_control_effect',
               'indirect_treated_effect', 'indirect_control_effect', 'n_non_trimmed', 'duration']
    res_df = pd.DataFrame(res_list, columns=columns)
    res_df.to_csv('results/simulations/20220118_simulations_isotonic.csv', index=False, sep='\t')
