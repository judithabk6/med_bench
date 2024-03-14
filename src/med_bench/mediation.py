"""
the objective of this script is to implement estimators for mediation in
causal inference, simulate data, and evaluate and compare estimators
"""

# first step, run r code to have the original implementation by Huber
# using rpy2 to have the same data in R and python...

import numpy as np
import pandas as pd
# import rpy2.robjects as robjects
# import rpy2.robjects.packages as rpackages
from numpy.random import default_rng
# from rpy2.robjects import numpy2ri, pandas2ri
from scipy import stats
from scipy.special import expit
from scipy.stats import bernoulli
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from .utils.nuisances import (_estimate_conditional_mean_outcome,
                              _estimate_cross_conditional_mean_outcome,
                              _estimate_cross_conditional_mean_outcome_nesting,
                              _estimate_mediator_density,
                              _estimate_treatment_probabilities,
                              _get_classifier, _get_regressor)
from .utils.utils import r_dependency_required
# import warnings

# if check_r_dependencies():
    # from .utils.utils import _convert_array_to_R
#     import rpy2.robjects as robjects
#     import rpy2.robjects.packages as rpackages
#     from rpy2.robjects import numpy2ri, pandas2ri

#     pandas2ri.activate()
#     numpy2ri.activate()

    # causalweight = rpackages.importr('causalweight')
    # mediation = rpackages.importr('mediation')
    # Rstats = rpackages.importr('stats')
    # base = rpackages.importr('base')
    # grf = rpackages.importr('grf')
    # plmed = rpackages.importr('plmed')


ALPHAS = np.logspace(-5, 5, 8)
CV_FOLDS = 5
TINY = 1.e-12


def mediation_IPW(y, t, m, x, trim, regularization=True, forest=False,
                  crossfit=0, clip=0.01, calibration='sigmoid'):
    """
    IPW estimator presented in
    HUBER, Martin. Identifying causal mechanisms (primarily) based on inverse
    probability weighting. Journal of Applied Econometrics, 2014,
    vol. 29, no 6, p. 920-943.

    Parameters
    ----------
    y : array-like, shape (n_samples)
            outcome value for each unit, continuous

    t : array-like, shape (n_samples)
            treatment value for each unit, binary

    m : array-like, shape (n_samples, n_features_mediator)
            mediator value for each unit, can be continuous or binary, and
            multidimensional

    x : array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    trim : float
            Trimming rule for discarding observations with extreme propensity
            scores. In the absence of post-treatment confounders (w=NULL),
            observations with Pr(D=1|M,X)<trim or Pr(D=1|M,X)>(1-trim) are
            dropped. In the presence of post-treatment confounders
            (w is defined), observations with Pr(D=1|M,W,X)<trim or
            Pr(D=1|M,W,X)>(1-trim) are dropped.

    regularization : boolean, default=True
                   whether to use regularized models (logistic or
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5

    forest : boolean, default=False
            whether to use a random forest model to estimate the propensity
            scores instead of logistic regression

    crossfit : integer, default=0
             number of folds for cross-fitting

    clip : float, default=0.01
            limit to clip for numerical stability (min=clip, max=1-clip)

    calibration : str, default=sigmoid
            calibration mode; for example using a sigmoid function

    Returns
    -------
    float
            total effect
    float
            direct effect treated (\theta(1))
    float
            direct effect nontreated (\theta(0))
    float
            indirect effect treated (\delta(1))
    float
            indirect effect untreated (\delta(0))
    int
            number of used observations (non trimmed)
    """
    # estimate propensities
    classifier_t_x = _get_classifier(regularization, forest, calibration)
    classifier_t_xm = _get_classifier(regularization, forest, calibration)
    p_x, p_xm = _estimate_treatment_probabilities(t, m, x, crossfit,
                                                  classifier_t_x,
                                                  classifier_t_xm)

   # trimming. Following causal weight code, not sure I understand
    # why we trim only on p_xm and not on p_x
    ind = ((p_xm > trim) & (p_xm < (1 - trim)))
    y, t, p_x, p_xm = y[ind], t[ind], p_x[ind], p_xm[ind]

    # note on the names, ytmt' = Y(t, M(t')), the treatment needs to be
    # binary but not the mediator
    p_x = np.clip(p_x, clip, 1 - clip)
    p_xm = np.clip(p_xm, clip, 1 - clip)

    # importance weighting
    y1m1 = np.sum(y * t / p_x) / np.sum(t / p_x)
    y1m0 = np.sum(y * t * (1 - p_xm) / (p_xm * (1 - p_x))) /\
        np.sum(t * (1 - p_xm) / (p_xm * (1 - p_x)))
    y0m0 = np.sum(y * (1 - t) / (1 - p_x)) /\
        np.sum((1 - t) / (1 - p_x))
    y0m1 = np.sum(y * (1 - t) * p_xm / ((1 - p_xm) * p_x)) /\
        np.sum((1 - t) * p_xm / ((1 - p_xm) * p_x))

    return(y1m1 - y0m0,
           y1m1 - y0m1,
           y1m0 - y0m0,
           y1m1 - y1m0,
           y0m1 - y0m0,
           np.sum(ind))


def mediation_coefficient_product(y, t, m, x, interaction=False,
                                  regularization=True):
    """
    found an R implementation https://cran.r-project.org/package=regmedint

    implements very simple model of mediation
    Y ~ X + T + M
    M ~ X + T
    estimation method is product of coefficients

    Parameters
    ----------
    y : array-like, shape (n_samples)
            outcome value for each unit, continuous

    t : array-like, shape (n_samples)
            treatment value for each unit, binary

    m : array-like, shape (n_samples)
            mediator value for each unit, can be continuous or binary, and
            is necessary in 1D

    x : array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    interaction : boolean, default=False
                whether to include interaction terms in the model
                not implemented here, just for compatibility of signature
                function

    regularization : boolean, default=True
                   whether to use regularized models (logistic or
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5

    Returns
    -------
    float
            total effect
    float
            direct effect treated (\theta(1))
    float
            direct effect nontreated (\theta(0))
    float
            indirect effect treated (\delta(1))
    float
            indirect effect untreated (\delta(0))
    int
            number of used observations (non trimmed)
    """
    if regularization:
        alphas = ALPHAS
    else:
        alphas = [TINY]
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)
    if len(t.shape) == 1:
        t = t.reshape(-1, 1)
    coef_t_m = np.zeros(m.shape[1])
    for i in range(m.shape[1]):
        m_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)\
            .fit(np.hstack((x, t)), m[:, i])
        coef_t_m[i] = m_reg.coef_[-1]
    y_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)\
        .fit(np.hstack((x, t, m)), y.ravel())

    # return total, direct and indirect effect
    direct_effect = y_reg.coef_[x.shape[1]]
    indirect_effect = sum(y_reg.coef_[x.shape[1] + 1:] * coef_t_m)
    return [direct_effect + indirect_effect,
            direct_effect,
            direct_effect,
            indirect_effect,
            indirect_effect,
            None]


def mediation_g_formula(y, t, m, x, interaction=False, forest=False,
                        crossfit=0, regularization=True,
                        calibration='sigmoid'):
    """
    Warning : m needs to be binary

    implementation of the g formula for mediation

    Parameters
    ----------
    y : array-like, shape (n_samples)
            outcome value for each unit, continuous

    t : array-like, shape (n_samples)
            treatment value for each unit, binary

    m : array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary and uni-
            dimensional

    x : array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    interaction : boolean, default=False
                whether to include interaction terms in the model
                interactions are terms XT, TM, MX

    forest : boolean, default=False
            whether to use a random forest model to estimate the propensity
            scores instead of logistic regression, and outcome model instead
            of linear regression

    crossfit : integer, default=0
             number of folds for cross-fitting

    regularization : boolean, default=True
                   whether to use regularized models (logistic or
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5

    calibration : str, default=sigmoid
            calibration mode; for example using a sigmoid function
    """
    # estimate mediator densities
    classifier_m = _get_classifier(regularization, forest, calibration)
    f_00x, f_01x, f_10x, f_11x, _, _ = _estimate_mediator_density(t, m, x, y,
                                                                  crossfit,
                                                                  classifier_m,
                                                                  interaction)

    # estimate conditional mean outcomes
    regressor_y = _get_regressor(regularization, forest)
    mu_00x, mu_01x, mu_10x, mu_11x, _, _ = (
        _estimate_conditional_mean_outcome(t, m, x, y, crossfit, regressor_y,
                                           interaction))

    # G computation
    direct_effect_i1 = mu_11x - mu_01x
    direct_effect_i0 = mu_10x - mu_00x
    n = len(y)
    direct_effect_treated = (direct_effect_i1 * f_11x
                             + direct_effect_i0 * f_10x).sum() / n
    direct_effect_control = (direct_effect_i1 * f_01x
                             + direct_effect_i0 * f_00x).sum() / n
    indirect_effect_i1 = f_11x - f_01x
    indirect_effect_i0 = f_10x - f_00x
    indirect_effect_treated = (indirect_effect_i1 * mu_11x
                               + indirect_effect_i0 * mu_10x).sum() / n
    indirect_effect_control = (indirect_effect_i1 * mu_01x
                               + indirect_effect_i0 * mu_00x).sum() / n
    total_effect = direct_effect_control + indirect_effect_treated

    return [total_effect,
            direct_effect_treated,
            direct_effect_control,
            indirect_effect_treated,
            indirect_effect_control,
            None]


def alternative_estimator(y, t, m, x, regularization=True):
    """
    presented in
    HUBER, Martin, LECHNER, Michael, et MELLACE, Giovanni.
    The finite sample performance of estimators for mediation analysis under
    sequential conditional independence.
    Journal of Business & Economic Statistics, 2016, vol. 34, no 1, p. 139-160.
    section 3.2.2

    Parameters
    ----------
    y : array-like, shape (n_samples)
            outcome value for each unit, continuous

    t : array-like, shape (n_samples)
            treatment value for each unit, binary

    m : array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary
            and unidimensional

    x : array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    regularization : boolean, default=True
                   whether to use regularized models (logistic or
                   linear regression). If True, cross-validation is used
                   to chose among 8 potential log-spaced values between
                   1e-5 and 1e5
    """
    if regularization:
        alphas = ALPHAS
    else:
        alphas = [TINY]
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(m.shape) == 1:
        m = m.reshape(-1, 1)
    treated = (t == 1)

    # computation of direct effect
    y_treated_reg_m = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(
        np.hstack((x[treated], m[treated])), y[treated])
    y_ctrl_reg_m = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(
        np.hstack((x[~treated], m[~treated])), y[~treated])
    xm = np.hstack((x, m))
    direct_effect = np.sum(y_treated_reg_m.predict(xm)
                           - y_ctrl_reg_m.predict(xm)) / len(y)

    # computation of total effect
    y_treated_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(
        x[treated], y[treated])
    y_ctrl_reg = RidgeCV(alphas=alphas, cv=CV_FOLDS).fit(
        x[~treated], y[~treated])
    total_effect = np.sum(y_treated_reg.predict(x)
                          - y_ctrl_reg.predict(x)) / len(y)

    # computation of indirect effect
    indirect_effect = total_effect - direct_effect

    return [total_effect,
            direct_effect,
            direct_effect,
            indirect_effect,
            indirect_effect,
            None]


def mediation_multiply_robust(y, t, m, x, interaction=False, forest=False,
                              crossfit=0, clip=0.01, normalized=True,
                              regularization=True, calibration="sigmoid"):
    """
    Presented in Eric J. Tchetgen Tchetgen. Ilya Shpitser.
    "Semiparametric theory for causal mediation analysis: Efficiency bounds,
    multiple robustness and sensitivity analysis."
    Ann. Statist. 40 (3) 1816 - 1845, June 2012.
    https://doi.org/10.1214/12-AOS990

    Parameters
    ----------
    y : array-like, shape (n_samples)
        Outcome value for each unit, continuous

    t : array-like, shape (n_samples)
        Treatment value for each unit, binary

    m : array-like, shape (n_samples)
        Mediator value for each unit, binary and unidimensional

    x : array-like, shape (n_samples, n_features_covariates)
        Covariates value for each unit, continuous

    interaction : boolean, default=False
        Whether to include interaction terms in the model
        interactions are terms XT, TM, MX

    forest : boolean, default=False
        Whether to use a random forest model to estimate the propensity
        scores instead of logistic regression, and outcome model instead
        of linear regression

    crossfit : integer, default=0
        Number of folds for cross-fitting. If crossfit<2, no cross-fitting is
        applied

    clip : float, default=0.01
        Limit to clip p_x and f_mtx for numerical stability (min=clip,
        max=1-clip)

    normalized : boolean, default=True
        Normalizes the inverse probability-based weights so they add up to 1,
        as described in "Identifying causal mechanisms (primarily) based on
        inverse probability weighting", Huber (2014),
        https://doi.org/10.1002/jae.2341

    regularization : boolean, default=True
        Whether to use regularized models (logistic or linear regression).
        If True, cross-validation is used to chose among 8 potential
        log-spaced values between 1e-5 and 1e5

    calibration : str, default="sigmoid"
        Which calibration method to use.
        Implemented calibration methods are "sigmoid" and "isotonic".


    Returns
    -------
    total : float
        Average total effect.
    direct1 : float
        Direct effect on the exposed.
    direct0 : float
        Direct effect on the unexposed,
    indirect1 : float
        Indirect effect on the exposed.
    indirect0 : float
        Indirect effect on the unexposed.
    n_discarded : int
        Number of discarded samples due to trimming.


    Raises
    ------
    ValueError
        - If t or y are multidimensional.
        - If x, t, m, or y don't have the same length.
        - If m is not binary.
    """
    # Format checking
    if len(y) != len(y.ravel()):
        raise ValueError("Multidimensional y is not supported")
    if len(t) != len(t.ravel()):
        raise ValueError("Multidimensional t is not supported")
    if len(m) != len(m.ravel()):
        raise ValueError("Multidimensional m is not supported")

    n = len(y)
    if len(x.shape) == 1:
        x.reshape(n, 1)
    if len(m.shape) == 1:
        m.reshape(n, 1)

    dim_m = m.shape[1]
    if n * dim_m != sum(m.ravel() == 1) + sum(m.ravel() == 0):
        raise ValueError("m is not binary")

    y = y.ravel()
    t = t.ravel()
    m = m.ravel()
    if n != len(x) or n != len(m) or n != len(t):
        raise ValueError("Inputs don't have the same number of observations")

    # estimate propensities
    classifier_t_x = _get_classifier(regularization, forest, calibration)
    p_x, _ = _estimate_treatment_probabilities(t, m, x, crossfit,
                                               classifier_t_x,
                                               clone(classifier_t_x))

    # estimate mediator densities
    classifier_m = _get_classifier(regularization, forest, calibration)
    f_00x, f_01x, f_10x, f_11x, f_m0x, f_m1x = (
        _estimate_mediator_density(t, m, x, y, crossfit,
                                   classifier_m, interaction))
    f = f_00x, f_01x, f_10x, f_11x

    # estimate conditional mean outcomes
    regressor_y = _get_regressor(regularization, forest)
    regressor_cross_y = _get_regressor(regularization, forest)
    mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1 = (
        _estimate_cross_conditional_mean_outcome(t, m, x, y, crossfit,
                                                 regressor_y,
                                                 regressor_cross_y, f,
                                                 interaction))

    # clipping
    p_x_clip = p_x != np.clip(p_x, clip, 1 - clip)
    f_m0x_clip = f_m0x != np.clip(f_m0x, clip, 1 - clip)
    f_m1x_clip = f_m1x != np.clip(f_m1x, clip, 1 - clip)
    clipped = p_x_clip + f_m0x_clip + f_m1x_clip

    var_name = ["t", "y", "p_x", "f_m0x", "f_m1x", "mu_1mx", "mu_0mx"]
    var_name += ["E_mu_t1_t1", "E_mu_t0_t0", "E_mu_t1_t0", "E_mu_t0_t1"]
    n_discarded = 0
    for var in var_name:
        exec(f"{var} = {var}[~clipped]")
    n_discarded += np.sum(clipped)

    # score computing
    if normalized:
        sum_score_m1 = np.mean(t / p_x)
        sum_score_m0 = np.mean((1 - t) / (1 - p_x))
        sum_score_t1m0 = np.mean((t / p_x) * (f_m0x / f_m1x))
        sum_score_t0m1 = np.mean((1 - t) / (1 - p_x) * (f_m1x / f_m0x))

        y1m1 = (t / p_x * (y - E_mu_t1_t1)) / sum_score_m1 + E_mu_t1_t1
        y0m0 = (((1 - t) / (1 - p_x) * (y - E_mu_t0_t0)) / sum_score_m0
                + E_mu_t0_t0)
        y1m0 = (
                ((t / p_x) * (f_m0x / f_m1x) * (y - mu_1mx)) / sum_score_t1m0
                + ((1 - t) / (1 - p_x) * (mu_1mx - E_mu_t1_t0)) / sum_score_m0
                + E_mu_t1_t0
        )
        y0m1 = (
                ((1 - t) / (1 - p_x) * (f_m1x / f_m0x) * (y - mu_0mx))
                / sum_score_t0m1 + t / p_x * (
                            mu_0mx - E_mu_t0_t1) / sum_score_m1
                + E_mu_t0_t1
        )
    else:
        y1m1 = t / p_x * (y - E_mu_t1_t1) + E_mu_t1_t1
        y0m0 = (1 - t) / (1 - p_x) * (y - E_mu_t0_t0) + E_mu_t0_t0
        y1m0 = (
                (t / p_x) * (f_m0x / f_m1x) * (y - mu_1mx)
                + (1 - t) / (1 - p_x) * (mu_1mx - E_mu_t1_t0)
                + E_mu_t1_t0
        )
        y0m1 = (
                (1 - t) / (1 - p_x) * (f_m1x / f_m0x) * (y - mu_0mx)
                + t / p_x * (mu_0mx - E_mu_t0_t1)
                + E_mu_t0_t1
        )

    # effects computing
    total = np.mean(y1m1 - y0m0)
    direct1 = np.mean(y1m1 - y0m1)
    direct0 = np.mean(y1m0 - y0m0)
    indirect1 = np.mean(y1m1 - y1m0)
    indirect0 = np.mean(y0m1 - y0m0)

    return total, direct1, direct0, indirect1, indirect0, n_discarded


@r_dependency_required(['mediation', 'stats', 'base'])
def r_mediate(y, t, m, x, interaction=False):
    """
    This function calls the R function mediate from the package mediation
    (https://cran.r-project.org/package=mediation)

    Parameters
    ----------
    y : array-like, shape (n_samples)
            outcome value for each unit, continuous

    t : array-like, shape (n_samples)
            treatment value for each unit, binary

    m : array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary and
            unidimensional

    x : array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    interaction : boolean, default=False
                whether to include interaction terms in the model
                interactions are terms XT, TM, MX
    """

    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import numpy2ri, pandas2ri

    pandas2ri.activate()
    numpy2ri.activate()

    mediation = rpackages.importr('mediation')
    Rstats = rpackages.importr('stats')
    base = rpackages.importr('base')

    m = m.ravel()
    var_names = [[y, 'y'],
                 [t, 't'],
                 [m, 'm'],
                 [x, 'x']]
    df_list = list()
    for var, name in var_names:
        if len(var.shape) > 1:
            var_dim = var.shape[1]
            col_names = ['{}_{}'.format(name, i) for i in range(var_dim)]
            sub_df = pd.DataFrame(var, columns=col_names)
        else:
            sub_df = pd.DataFrame(var, columns=[name])
        df_list.append(sub_df)
        df = pd.concat(df_list, axis=1)
    m_features = [c for c in df.columns if ('y' not in c) and ('m' not in c)]
    y_features = [c for c in df.columns if ('y' not in c)]
    if not interaction:
        m_formula = 'm ~ ' + ' + '.join(m_features)
        y_formula = 'y ~ ' + ' + '.join(y_features)
    else:
        m_formula = 'm ~ ' + ' + '.join(m_features +
                                        [':'.join(p) for p in
                                         combinations(m_features, 2)])
        y_formula = 'y ~ ' + ' + '.join(y_features +
                                        [':'.join(p) for p in
                                         combinations(y_features, 2)])
    robjects.globalenv['df'] = df
    mediator_model = Rstats.lm(m_formula, data=base.as_symbol('df'))
    outcome_model = Rstats.lm(y_formula, data=base.as_symbol('df'))
    res = mediation.mediate(mediator_model, outcome_model, treat='t',
                            mediator='m', boot=True, sims=1)

    relevant_variables = ['tau.coef', 'z1', 'z0', 'd1', 'd0']
    to_return = [np.array(res.rx2(v))[0] for v in relevant_variables]
    return to_return + [None]


@r_dependency_required(['plmed', 'base'])
def r_mediation_g_estimator(y, t, m, x):
    """
    This function calls the R G-estimator from the package plmed
    (https://github.com/ohines/plmed)
    """

    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import numpy2ri, pandas2ri

    pandas2ri.activate()
    numpy2ri.activate()

    plmed = rpackages.importr('plmed')
    base = rpackages.importr('base')

    m = m.ravel()
    var_names = [[y, 'y'],
                 [t, 't'],
                 [m, 'm'],
                 [x, 'x']]
    df_list = list()
    for var, name in var_names:
        if len(var.shape) > 1:
            var_dim = var.shape[1]
            col_names = ['{}_{}'.format(name, i) for i in range(var_dim)]
            sub_df = pd.DataFrame(var, columns=col_names)
        else:
            sub_df = pd.DataFrame(var, columns=[name])
        df_list.append(sub_df)
        df = pd.concat(df_list, axis=1)
    m_features = [c for c in df.columns if ('x' in c)]
    y_features = [c for c in df.columns if ('x' in c)]
    t_features = [c for c in df.columns if ('x' in c)]
    m_formula = 'm ~ ' + ' + '.join(m_features)
    y_formula = 'y ~ ' + ' + '.join(y_features)
    t_formula = 't ~ ' + ' + '.join(t_features)
    robjects.globalenv['df'] = df
    res = plmed.G_estimation(t_formula,
                             m_formula,
                             y_formula,
                             exposure_family='binomial',
                             data=base.as_symbol('df'))
    direct_effect = res.rx2('coef')[0]
    indirect_effect = res.rx2('coef')[1]
    return [direct_effect + indirect_effect,
            direct_effect,
            direct_effect,
            indirect_effect,
            indirect_effect,
            None]


@r_dependency_required(['causalweight', 'base'])
def r_mediation_DML(y, t, m, x, trim=0.05, order=1):
    """
    This function calls the R Double Machine Learning estimator from the
    package causalweight (https://cran.r-project.org/web/packages/causalweight)

    Parameters
    ----------
    y : array-like, shape (n_samples)
            outcome value for each unit, continuous

    t : array-like, shape (n_samples)
            treatment value for each unit, binary

    m : array-like, shape (n_samples, n_features_mediator)
            mediator value for each unit, can be continuous or binary, and
            multi-dimensional

    x : array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    trim : float, default=0.05
            Trimming rule for discarding observations with extreme
            conditional treatment or mediator probabilities
            (or products thereof). Observations with (products of)
            conditional probabilities that are smaller than trim in any
            denominator of the potential outcomes are dropped.

    order : integer, default=1
            If set to an integer larger than 1, then polynomials of that
            order and interactions using the power series) rather than the
            original control variables are used in the estimation of any
            conditional probability or conditional mean outcome.
            Polynomials/interactions are created using the Generate.
            Powers command of the LARF package.
    """
    
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import numpy2ri, pandas2ri
    from .utils.utils import _convert_array_to_R

    pandas2ri.activate()
    numpy2ri.activate()

    causalweight = rpackages.importr('causalweight')
    base = rpackages.importr('base')

    x_r, t_r, m_r, y_r = [base.as_matrix(_convert_array_to_R(uu)) for uu in
                          (x, t, m, y)]
    res = causalweight.medDML(y_r, t_r, m_r, x_r, trim=trim, order=order)
    raw_res_R = np.array(res.rx2('results'))
    ntrimmed = res.rx2('ntrimmed')[0]
    return list(raw_res_R[0, :5]) + [ntrimmed]


def mediation_DML(y, t, m, x, forest=False, crossfit=0, trim=0.05,
                  normalized=True, regularization=True, random_state=None,
                  calibration=None):
    """
    Python implementation of Double Machine Learning procedure, as described
    in :
    Helmut Farbmacher and others, Causal mediation analysis with double
    machine learning,
    The Econometrics Journal, Volume 25, Issue 2, May 2022, Pages 277–300,
    https://doi.org/10.1093/ectj/utac003

    Parameters
    ----------

    y : array-like, shape (n_samples)
        Outcome value for each unit.

    t : array-like, shape (n_samples)
        Treatment value for each unit.

    m : array-like, shape (n_samples, n_features_mediator)
        Mediator value for each unit, multidimensional or continuous.

    x : array-like, shape (n_samples, n_features_covariates)
        Covariates value for each unit, multidimensional or continuous.

    forest : boolean, default=False
        Whether to use a random forest model to estimate the propensity
        scores instead of logistic regression, and outcome model instead
        of linear regression.

    crossfit : int, default=0
        Number of folds for cross-fitting.

    trim : float, default=0.05
        Trimming treshold for discarding observations with extreme probability.

    normalized : boolean, default=True
        Normalizes the inverse probability-based weights so they add up to 1,
        as described in "Identifying causal mechanisms (primarily) based on
        inverse probability weighting",
        Huber (2014), https://doi.org/10.1002/jae.2341

    regularization : boolean, default=True
        Whether to use regularized models (logistic or linear regression).
        If True, cross-validation is used to chose among 8 potential
        log-spaced values between 1e-5 and 1e5.

    random_state : int, default=None
        LogisticRegression random state instance.

    calibration : {None, "sigmoid", "isotonic"}, default=None
        Whether to add a calibration step for the classifier used to estimate
        the treatment propensity score and P(T|M,X). "None" means no
        calibration.
        Calibration ensures the output of the [predict_proba]
        (https://scikit-learn.org/stable/glossary.html#term-predict_proba)
        method can be directly interpreted as a confidence level.
        Implemented calibration methods are "sigmoid" and "isotonic".

    Returns
    -------
    total : float
        Average total effect.
    direct1 : float
        Direct effect on the exposed.
    direct0 : float
        Direct effect on the unexposed,
    indirect1 : float
        Indirect effect on the exposed.
    indirect0 : float
        Indirect effect on the unexposed.
    n_discarded : int
        Number of discarded samples due to trimming.

    Raises
    ------
    ValueError
        - If t or y are multidimensional.
        - If x, t, m, or y don't have the same length.
    """
    # check format
    if len(y) != len(y.ravel()):
        raise ValueError("Multidimensional y is not supported")

    if len(t) != len(t.ravel()):
        raise ValueError("Multidimensional t is not supported")

    n = len(y)
    t = t.ravel()
    y = y.ravel()

    if n != len(x) or n != len(m) or n != len(t):
        raise ValueError("Inputs don't have the same number of observations")

    if len(x.shape) == 1:
        x.reshape(n, 1)

    if len(m.shape) == 1:
        m.reshape(n, 1)

    nobs = 0


    var_name = [
        "p_x",
        "p_xm",
        "mu_1mx",
        "mu_0mx",
        "E_mu_t1_t0",
        "E_mu_t0_t1",
        "E_mu_t1_t1",
        "E_mu_t0_t0",
    ]

    # estimate propensities
    classifier_t_x = _get_classifier(regularization, forest, calibration)
    classifier_t_xm = _get_classifier(regularization, forest, calibration)
    p_x, p_xm = _estimate_treatment_probabilities(t, m, x, crossfit,
                                                  classifier_t_x,
                                                  classifier_t_xm)

    # estimate conditional mean outcomes
    regressor_y = _get_regressor(regularization, forest)
    regressor_cross_y = _get_regressor(regularization, forest)

    mu_0mx, mu_1mx, E_mu_t0_t0, E_mu_t0_t1, E_mu_t1_t0, E_mu_t1_t1 = (
        _estimate_cross_conditional_mean_outcome_nesting(t, m, x, y, crossfit,
                                                         regressor_y,
                                                         regressor_cross_y))

    # trimming
    not_trimmed = (
        (((1 - p_xm) * p_x) >= trim)
        * ((1 - p_x) >= trim)
        * (p_x >= trim)
        * ((p_xm * (1 - p_x)) >= trim)
    )
    for var in var_name:
        exec(f"{var} = {var}[not_trimmed]")
    nobs = np.sum(not_trimmed)

    # score computing
    if normalized:
        sum_score_m1 = np.mean(t / p_x)
        sum_score_m0 = np.mean((1 - t) / (1 - p_x))
        sum_score_t1m0 = np.mean(t * (1 - p_xm) / (p_xm * (1 - p_x)))
        sum_score_t0m1 = np.mean((1 - t) * p_xm / ((1 - p_xm) * p_x))
        y1m1 = (t / p_x * (y - E_mu_t1_t1)) / sum_score_m1 + E_mu_t1_t1
        y0m0 = (((1 - t) / (1 - p_x) * (y - E_mu_t0_t0)) / sum_score_m0
                + E_mu_t0_t0)
        y1m0 = (
            (t * (1 - p_xm) / (p_xm * (1 - p_x)) * (y - mu_1mx))
            / sum_score_t1m0 + ((1 - t) / (1 - p_x) * (mu_1mx - E_mu_t1_t0))
            / sum_score_m0 + E_mu_t1_t0
        )
        y0m1 = (
            ((1 - t) * p_xm / ((1 - p_xm) * p_x) * (y - mu_0mx))
            / sum_score_t0m1
            + (t / p_x * (mu_0mx - E_mu_t0_t1)) / sum_score_m1
            + E_mu_t0_t1
        )
    else:
        y1m1 = t / p_x * (y - E_mu_t1_t1) + E_mu_t1_t1
        y0m0 = (1 - t) / (1 - p_x) * (y - E_mu_t0_t0) + E_mu_t0_t0
        y1m0 = (
            t * (1 - p_xm) / (p_xm * (1 - p_x)) * (y - mu_1mx)
            + (1 - t) / (1 - p_x) * (mu_1mx - E_mu_t1_t0)
            + E_mu_t1_t0
        )
        y0m1 = (
            (1 - t) * p_xm / ((1 - p_xm) * p_x) * (y - mu_0mx)
            + t / p_x * (mu_0mx - E_mu_t0_t1)
            + E_mu_t0_t1
        )

    # mean score computing
    my1m1 = np.mean(y1m1)
    my0m0 = np.mean(y0m0)
    my1m0 = np.mean(y1m0)
    my0m1 = np.mean(y0m1)

    # effects computing
    total = my1m1 - my0m0
    direct1 = my1m1 - my0m1
    direct0 = my1m0 - my0m0
    indirect1 = my1m1 - my1m0
    indirect0 = my0m1 - my0m0
    return total, direct1, direct0, indirect1, indirect0, n - nobs