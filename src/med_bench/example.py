from numpy.random import default_rng
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import train_test_split

from med_bench.mediation import (mediation_IPW, mediation_coefficient_product, mediation_dml,
                                 mediation_g_formula, mediation_multiply_robust)
from med_bench.estimation.mediation_coefficient_product import CoefficientProduct
from med_bench.estimation.mediation_dml import DoubleMachineLearning
from med_bench.estimation.mediation_g_computation import GComputation
from med_bench.estimation.mediation_ipw import ImportanceWeighting
from med_bench.estimation.mediation_mr import MultiplyRobust
from med_bench.get_simulated_data import simulate_data
from med_bench.nuisances.utils import _get_regularization_parameters
from med_bench.utils.constants import CV_FOLDS


if __name__ == "__main__":
    print("get simulated data")
    (x, t, m, y,
     theta_1_delta_0, theta_1, theta_0, delta_1, delta_0,
     p_t, th_p_t_mx) = simulate_data(n=1000, rg=default_rng(321), dim_x=5)

    (x_train, x_test, t_train, t_test,
        m_train, m_test, y_train, y_test) = train_test_split(x, t, m, y, test_size=0.33, random_state=42)

    cs, alphas = _get_regularization_parameters(regularization=True)

    clf = RandomForestClassifier(
        random_state=42, n_estimators=100, min_samples_leaf=10)

    clf2 = LogisticRegressionCV(random_state=42, Cs=cs, cv=CV_FOLDS)

    reg = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=10, random_state=42)

    reg2 = RidgeCV(alphas=alphas, cv=CV_FOLDS)

    # Step 4: Define estimators (modularized and non-modularized)
    estimators = {
        "CoefficientProduct": {
            "modular": CoefficientProduct(
                regressor=reg, classifier=clf, regularize=True
            ),
            "non_modular": mediation_coefficient_product
        },
        "DoubleMachineLearning": {
            "modular": DoubleMachineLearning(
                clip=1e-6, trim=0.05, normalized=True, regressor=reg2, classifier=clf2
            ),
            "non_modular": mediation_dml
        },
        "GComputation": {
            "modular": GComputation(
                regressor=reg2, classifier=CalibratedClassifierCV(
                    clf2, method="sigmoid")
            ),
            "non_modular": mediation_g_formula
        },
        "ImportanceWeighting": {
            "modular": ImportanceWeighting(
                clip=1e-6, trim=0.01, regressor=reg2, classifier=CalibratedClassifierCV(clf2, method="sigmoid")
            ),
            "non_modular": mediation_IPW
        },
        "MultiplyRobust": {
            "modular": MultiplyRobust(
                clip=1e-6, ratio="propensities", normalized=True, regressor=reg2,
                classifier=CalibratedClassifierCV(clf2, method="sigmoid")
            ),
            "non_modular": mediation_multiply_robust
        }
    }

    # Step 5: Initialize results DataFrame
    results = []

    # Step 6: Iterate over each estimator
    for estimator_name, estimator_dict in estimators.items():
        # Non-Modularized Estimation
        # Check if non-modular is a function
        if callable(estimator_dict["non_modular"]):
            (total_effect, direct_effect1, direct_effect2, indirect_effect1, indirect_effect2, _) = estimator_dict["non_modular"](
                y, t, m, x)

            results.append({
                "Estimator": estimator_name,
                "Method": "Non-Modularized",
                "Total Effect": total_effect,
                "Direct Effect (Treated)": direct_effect1,
                "Direct Effect (Control)": direct_effect2,
                "Indirect Effect (Treated)": indirect_effect1,
                "Indirect Effect (Control)": indirect_effect2,
            })

        # Modularized Estimation
        modular_estimator = estimator_dict["modular"]
        modular_estimator.fit(t_train, m_train, x_train, y_train)
        causal_effects = modular_estimator.estimate(
            t_test, m_test, x_test, y_test)

        # Append modularized results
        results.append({
            "Estimator": estimator_name,
            "Method": "Modularized",
            "Total Effect": causal_effects['total_effect'],
            "Direct Effect (Treated)": causal_effects['direct_effect_treated'],
            "Direct Effect (Control)": causal_effects['direct_effect_control'],
            "Indirect Effect (Treated)": causal_effects['indirect_effect_treated'],
            "Indirect Effect (Control)": causal_effects['indirect_effect_control'],
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display or save the DataFrame
    print(results_df)
