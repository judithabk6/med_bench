"""
the objective of this module is to implement estimators for mediation in
causal inference, those used directly from their R implementation
"""


import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri, numpy2ri
import numpy as np
from scipy import stats
import pandas as pd

from itertools import combinations

pandas2ri.activate()
numpy2ri.activate()

causalweight = rpackages.importr('causalweight')
mediation = rpackages.importr('mediation')
Rstats = rpackages.importr('stats')
base = rpackages.importr('base')
grf = rpackages.importr('grf')
plmed = rpackages.importr('plmed')



def r_mediate(y, t, m, x, interaction=False):
    """
    This function calls the R function mediate from the package mediation
    (https://cran.r-project.org/package=mediation)
    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples)
            mediator value for each unit, here m is necessary binary and uni-
            dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    interaction boolean, default=False
                whether to include interaction terms in the model
                interactions are terms XT, TM, MX
    """
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


def g_estimator(y, t, m, x):
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




def medDML(y, t, m, x, trim=0.05, order=1):
    """
    y       array-like, shape (n_samples)
            outcome value for each unit, continuous

    t       array-like, shape (n_samples)
            treatment value for each unit, binary

    m       array-like, shape (n_samples, n_features_mediator)
            mediator value for each unit, can be continuous or binary, and
            multi-dimensional

    x       array-like, shape (n_samples, n_features_covariates)
            covariates (potential confounders) values

    trim    float
            Trimming rule for discarding observations with extreme
            conditional treatment or mediator probabilities
            (or products thereof). Observations with (products of)
            conditional probabilities that are smaller than trim in any
            denominator of the potential outcomes are dropped.
            Default is 0.05.

    order   integer
            If set to an integer larger than 1, then polynomials of that
            order and interactions using the power series) rather than the
            original control variables are used in the estimation of any
            conditional probability or conditional mean outcome.
            Polynomials/interactions are created using the Generate.
            Powers command of the LARF package.
    """
    x_r, t_r, m_r, y_r = [base.as_matrix(_convert_array_to_R(uu)) for uu in
                          (x, t, m, y)]
    res = causalweight.medDML(y_r, t_r, m_r, x_r, trim=trim, order=order)
    raw_res_R = np.array(res.rx2('results'))
    ntrimmed = res.rx2('ntrimmed')[0]
    return list(raw_res_R[0, :5]) + [ntrimmed]


def _convert_array_to_R(x):
    """
    converts a numpy array to a R matrix or vector
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.sum(a == np.array(_convert_array_to_R(a)))
    6
    """
    if len(x.shape) == 1:
        return robjects.FloatVector(x)
    elif len(x.shape) == 2:
        return robjects.r.matrix(robjects.FloatVector(x.ravel()),
                                 nrow=x.shape[0], byrow='TRUE')
