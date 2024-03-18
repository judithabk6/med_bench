import numpy as np
from numpy.random import default_rng
from scipy import stats
from scipy.special import expit


def simulate_data(n,
                  rg,
                  mis_spec_m=False,
                  mis_spec_y=False,
                  dim_x=1,
                  dim_m=1,
                  seed=None,
                  type_m='binary',
                  sigma_y=0.5,
                  sigma_m=0.5,
                  beta_t_factor=1,
                  beta_m_factor=1):
    """Simulate data for mediation analysis

    Parameters
    ----------
    n:  :obj:`int`,
        Number of samples to generate.

    rg: RandomState instance,
        Controls the pseudo random number generator used to generate the
        data at fit time.

    mis_spec_m: obj:`bool`, 
        Whether the mediator generation is misspecified or not
        defaults to False

    mis_spec_y: obj:`bool`, 
        Whether the output model is misspecified or not
        defaults to False

    dim_x: :obj:`int`, optional,
        Number of covariates in the input.
        Defaults to 1

    dim_m: :obj:`int`, optional,
        Number of mediatiors to generate.
        Defaults to 1

    seed: :obj:`int` or None, optional,
        Controls the pseudo random number generator used to generate the
        coefficients of the model.
        Pass an int for reproducible output across multiple function calls.
        Defaults to None

    type_m: :obj:`str`,
        Whether the mediator is binary or continuous
        Defaults to 'binary',

    sigma_y: :obj:`float`,
        noise variance on outcome
        Defaults to 0.5,

    sigma_m :obj:`float`,
        noise variance on mediator
        Defaults to 0.5,

    beta_t_factor: :obj:`float`,
        scaling factor on treatment effect,
        Defaults to 1,

    beta_m_factor: :obj:`float`,
        scaling factor on mediator,
        Defaults to 1,

    returns
    -------
    x: ndarray of shape (n, dim_x)
        the simulated covariates

    t: ndarray of shape (n, 1)
        the simulated treatment

    m: ndarray of shape (n, dim_m)
        the simulated mediators

    y: ndarray of shape (n, 1)
        the simulated outcome

    total:  :obj:`float`,
        the total simulated effect

    theta_1: :obj:`float`,
        the natural direct effect on the treated, 

    theta_0: :obj:`float`,
        the natural direct effect on the untreated, 

    delta_1: :obj:`float`,
        the natural indirect effect on the treated, 

    delta_0: :obj:`float`,
        the natural indirect effect on the untreated, 

    p_t: ndarray of shape (n, 1),
        Propensity score

    th_p_t_mx: ndarray of shape (n, 1),
        overlap

    """
    rg_coef = default_rng(seed)
    x = rg.standard_normal(n * dim_x).reshape((n, dim_x))
    alphas = np.ones(dim_x) / dim_x
    p_t = expit(alphas.dot(x.T))
    t = rg.binomial(1, p_t, n).reshape(-1, 1)
    t0 = np.zeros((n, 1))
    t1 = np.ones((n, 1))

    # generate the mediator M
    beta_x = rg_coef.standard_normal((dim_x, dim_m)) * 1 / (dim_m * dim_x)
    beta_t = np.ones((1, dim_m)) * beta_t_factor
    if mis_spec_m:
        beta_xt = rg_coef.standard_normal((dim_x, dim_m)) * 1 / (dim_m * dim_x)
    else:
        beta_xt = np.zeros((dim_x, dim_m))

    if type_m == 'binary':
        p_m0 = expit(x.dot(beta_x) + beta_t * t0 + x.dot(beta_xt) * t0)
        p_m1 = expit(x.dot(beta_x) + beta_t * t1 + x.dot(beta_xt) * t1)
        pre_m = rg.random(n)
        m0 = ((pre_m < p_m0.ravel()) * 1).reshape(-1, 1)
        m1 = ((pre_m < p_m1.ravel()) * 1).reshape(-1, 1)
        m_2d = np.hstack((m0, m1))
        m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
    else:
        random_noise = sigma_m * rg.standard_normal((n, dim_m))
        m0 = x.dot(beta_x) + t0.dot(beta_t) + t0 * \
            (x.dot(beta_xt)) + random_noise
        m1 = x.dot(beta_x) + t1.dot(beta_t) + t1 * \
            (x.dot(beta_xt)) + random_noise
        m = x.dot(beta_x) + t.dot(beta_t) + t * (x.dot(beta_xt)) + random_noise

    # generate the outcome Y
    gamma_m = np.ones((dim_m, 1)) * 0.5 / dim_m * beta_m_factor
    gamma_x = np.ones((dim_x, 1)) / dim_x**2
    gamma_t = 1.2
    if mis_spec_y:
        gamma_t_m = np.ones((dim_m, 1)) * 0.5 / dim_m
    else:
        gamma_t_m = np.zeros((dim_m, 1))

    y = x.dot(gamma_x) + gamma_t * t + m.dot(gamma_m) + \
        m.dot(gamma_t_m) * t + sigma_y * rg.standard_normal((n, 1))

    # Compute differents types of effects
    if type_m == 'binary':
        theta_1 = gamma_t + gamma_t_m * np.mean(p_m1)
        theta_0 = gamma_t + gamma_t_m * np.mean(p_m0)
        delta_1 = np.mean(
            (p_m1 - p_m0) * (gamma_m.flatten() + gamma_t_m.dot(t1.T)))
        delta_0 = np.mean(
            (p_m1 - p_m0) * (gamma_m.flatten() + gamma_t_m.dot(t0.T)))
    else:
        # to do mean(m1) pour avoir un vecteur de taille dim_m
        theta_1 = gamma_t + gamma_t_m.T.dot(np.mean(m1, axis=0))
        theta_0 = gamma_t + gamma_t_m.T.dot(np.mean(m0, axis=0))
        delta_1 = (gamma_t * t1 + m1.dot(gamma_m) + m1.dot(gamma_t_m) * t1 -
                   (gamma_t * t1 + m0.dot(gamma_m) + m0.dot(gamma_t_m) * t1)).mean()
        delta_0 = (gamma_t * t0 + m1.dot(gamma_m) + m1.dot(gamma_t_m) * t0 -
                   (gamma_t * t0 + m0.dot(gamma_m) + m0.dot(gamma_t_m) * t0)).mean()

    if type_m == 'binary':
        pre_pm = np.hstack((p_m0.reshape(-1, 1), p_m1.reshape(-1, 1)))
        pre_pm[m.ravel() == 0, :] = 1 - pre_pm[m.ravel() == 0, :]
        pm = pre_pm[:, 1].reshape(-1, 1)
    else:
        p_m0 = np.prod(stats.norm.pdf((m - x.dot(beta_x)) -
                       t0.dot(beta_t) - t0 * (x.dot(beta_xt)) / sigma_m), axis=1)
        p_m1 = np.prod(stats.norm.pdf((m - x.dot(beta_x)) -
                       t1.dot(beta_t) - t1 * (x.dot(beta_xt)) / sigma_m), axis=1)
        pre_pm = np.hstack((p_m0.reshape(-1, 1), p_m1.reshape(-1, 1)))
        pm = pre_pm[:, 1].reshape(-1, 1)

    px = np.prod(stats.norm.pdf(x), axis=1)

    pre_pt = np.hstack(((1-p_t).reshape(-1, 1), p_t.reshape(-1, 1)))
    double_px = np.hstack((px.reshape(-1, 1), px.reshape(-1, 1)))
    denom = np.sum(pre_pm * pre_pt * double_px, axis=1)
    num = pm.ravel() * p_t.ravel() * px.ravel()
    th_p_t_mx = num.ravel() / denom

    return (x,
            t,
            m,
            y,
            theta_1.flatten()[0] + delta_0.flatten()[0],
            theta_1.flatten()[0],
            theta_0.flatten()[0],
            delta_1.flatten()[0],
            delta_0.flatten()[0],
            p_t,
            th_p_t_mx)
