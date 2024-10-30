from numpy.random import default_rng
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from med_bench.estimation.mediation_coefficient_product import CoefficientProduct
from med_bench.get_simulated_data import simulate_data
from med_bench.nuisances.utils import _get_regularization_parameters
from med_bench.utils.constants import CV_FOLDS

if __name__ == "__main__":
    print("get simulated data")
    (x, t, m, y,
     theta_1_delta_0, theta_1, theta_0, delta_1, delta_0,
     p_t, th_p_t_mx) = simulate_data(n=1000, rg=default_rng(321))

    (x_train, x_test, t_train, t_test,
        m_train, m_test, y_train, y_test) = train_test_split(x, t, m, y, test_size=0.33, random_state=42)

    cs, alphas = _get_regularization_parameters(regularization=True)

    clf = RandomForestClassifier(
        random_state=42, n_estimators=100, min_samples_leaf=10)

    reg = RidgeCV(alphas=alphas, cv=CV_FOLDS)

    coef_prod_estimator = CoefficientProduct(
        mediator_type="binary", regressor=reg, classifier=clf, clip=0.01, trim=0.01, regularize=True)

    coef_prod_estimator.fit(t_train, m_train, x_train, y_train)
    causal_effects = coef_prod_estimator.estimate(
        t_test, m_test, x_test, y_test)

    r_risk_score = coef_prod_estimator.score(
        t_test, m_test, x_test, y_test, causal_effects['total_effect'])

    print('R risk score: {}'.format(r_risk_score))
    print('Total effect error: {}'.format(
        abs(causal_effects['total_effect']-theta_1_delta_0)))
    print('Direct effect error: {}'.format(
        abs(causal_effects['direct_effect_control']-theta_0)))
    print('Indirect effect error: {}'.format(
        abs(causal_effects['indirect_effect_treated']-delta_1)))
