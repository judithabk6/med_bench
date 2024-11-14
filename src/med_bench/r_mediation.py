"""
the objective of this script is to implement estimators for mediation in
causal inference, simulate data, and evaluate and compare estimators
"""

# first step, run r code to have the original implementation by Huber
# using rpy2 to have the same data in R and python...

import numpy as np
import pandas as pd

from .utils.utils import r_dependency_required, _check_input


@r_dependency_required(["mediation", "stats", "base"])
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

    mediation = rpackages.importr("mediation")
    Rstats = rpackages.importr("stats")
    base = rpackages.importr("base")

    # check input
    y, t, m, x = _check_input(y, t, m, x, setting="binary")
    m = m.ravel()

    var_names = [[y, "y"], [t, "t"], [m, "m"], [x, "x"]]
    df_list = list()
    for var, name in var_names:
        if len(var.shape) > 1:
            var_dim = var.shape[1]
            col_names = ["{}_{}".format(name, i) for i in range(var_dim)]
            sub_df = pd.DataFrame(var, columns=col_names)
        else:
            sub_df = pd.DataFrame(var, columns=[name])
        df_list.append(sub_df)
        df = pd.concat(df_list, axis=1)
    m_features = [c for c in df.columns if ("y" not in c) and ("m" not in c)]
    y_features = [c for c in df.columns if ("y" not in c)]
    if not interaction:
        m_formula = "m ~ " + " + ".join(m_features)
        y_formula = "y ~ " + " + ".join(y_features)
    else:
        m_formula = "m ~ " + " + ".join(
            m_features + [":".join(p) for p in combinations(m_features, 2)]
        )
        y_formula = "y ~ " + " + ".join(
            y_features + [":".join(p) for p in combinations(y_features, 2)]
        )
    robjects.globalenv["df"] = df
    mediator_model = Rstats.lm(m_formula, data=base.as_symbol("df"))
    outcome_model = Rstats.lm(y_formula, data=base.as_symbol("df"))
    res = mediation.mediate(
        mediator_model, outcome_model, treat="t", mediator="m", boot=True, sims=1
    )

    relevant_variables = ["tau.coef", "z1", "z0", "d1", "d0"]
    to_return = [np.array(res.rx2(v))[0] for v in relevant_variables]
    return to_return + [None]


@r_dependency_required(["plmed", "base"])
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

    plmed = rpackages.importr("plmed")
    base = rpackages.importr("base")

    # check input
    y, t, m, x = _check_input(y, t, m, x, setting="binary")
    m = m.ravel()

    var_names = [[y, "y"], [t, "t"], [m, "m"], [x, "x"]]
    df_list = list()
    for var, name in var_names:
        if len(var.shape) > 1:
            var_dim = var.shape[1]
            col_names = ["{}_{}".format(name, i) for i in range(var_dim)]
            sub_df = pd.DataFrame(var, columns=col_names)
        else:
            sub_df = pd.DataFrame(var, columns=[name])
        df_list.append(sub_df)
        df = pd.concat(df_list, axis=1)
    m_features = [c for c in df.columns if ("x" in c)]
    y_features = [c for c in df.columns if ("x" in c)]
    t_features = [c for c in df.columns if ("x" in c)]
    m_formula = "m ~ " + " + ".join(m_features)
    y_formula = "y ~ " + " + ".join(y_features)
    t_formula = "t ~ " + " + ".join(t_features)
    robjects.globalenv["df"] = df
    res = plmed.G_estimation(
        t_formula,
        m_formula,
        y_formula,
        exposure_family="binomial",
        data=base.as_symbol("df"),
    )
    direct_effect = res.rx2("coef")[0]
    indirect_effect = res.rx2("coef")[1]
    return (
        direct_effect + indirect_effect,
        direct_effect,
        direct_effect,
        indirect_effect,
        indirect_effect,
        None,
    )


@r_dependency_required(["causalweight", "base"])
def r_mediation_dml(y, t, m, x, trim=0.05, order=1):
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

    causalweight = rpackages.importr("causalweight")
    base = rpackages.importr("base")

    # check input
    y, t, m, x = _check_input(y, t, m, x, setting="multidimensional")

    x_r, t_r, m_r, y_r = [
        base.as_matrix(_convert_array_to_R(uu)) for uu in (x, t, m, y)
    ]
    res = causalweight.medDML(y_r, t_r, m_r, x_r, trim=trim, order=order)
    raw_res_R = np.array(res.rx2("results"))
    ntrimmed = res.rx2("ntrimmed")[0]
    return list(raw_res_R[0, :5]) + [ntrimmed]
