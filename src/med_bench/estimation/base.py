from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone, RegressorMixin, ClassifierMixin

from med_bench.utils.decorators import fitted
from med_bench.utils.scores import r_risk
from med_bench.utils.utils import _get_interactions


class Estimator:
    """General abstract class for causal mediation Estimator
    """
    __metaclass__ = ABCMeta
    def __init__(self, mediator_type : str, regressor : RegressorMixin, classifier : ClassifierMixin,
                  verbose : bool=True):
        """Initialize Estimator base class

        Parameters
        ----------
        mediator_type : str
            mediator type (binary or continuous)
        regressor : RegressorMixin
            Scikit-Learn Regressor used for mu estimation
        classifier : ClassifierMixin
            Scikit-Learn Classifier used for propensity estimation
        verbose : bool
            will print some logs if True
        """
        self.rng = np.random.RandomState(123)

        self.mediator_type = mediator_type

        # TBD inside an Issue
        self.regressor = regressor
        #self.regressor_params_dict = regressor_params_dict

        self.classifier = classifier
        #self.classifier_params_dict = classifier_params_dict

        self._verbose = verbose
        self._fitted = False


    @property
    def verbose(self):
        return self._verbose


    @abstractmethod
    def fit(self, t, m, x, y):
        """Fits nuisance parameters to data

        Parameters
        ----------
        t       array-like, shape (n_samples)
                treatment value for each unit, binary

        m       array-like, shape (n_samples)
                mediator value for each unit, here m is necessary binary and uni-
                dimensional

        x       array-like, shape (n_samples, n_features_covariates)
                covariates (potential confounders) values

        y       array-like, shape (n_samples)
                outcome value for each unit, continuous

        """
        pass


    @abstractmethod
    @fitted
    def estimate(self, t, m, x, y):
        """Estimates causal effect on data

        Parameters
        ----------
        t       array-like, shape (n_samples)
                treatment value for each unit, binary

        m       array-like, shape (n_samples)
                mediator value for each unit, here m is necessary binary and uni-
                dimensional

        x       array-like, shape (n_samples, n_features_covariates)
                covariates (potential confounders) values

        y       array-like, shape (n_samples)
                outcome value for each unit, continuous

        nuisances
        """
        pass


    def fit_score_nuisances(self, t, m, x, y, *args, **kwargs):
        """ Fits the score of the nuisance parameters
        """
        clf_param_grid = {}
        reg_param_grid = {}

        classifier_x = GridSearchCV(self.classifier, clf_param_grid)

        self._hat_e = classifier_x.fit(x, t.squeeze())

        regressor_y = GridSearchCV(self.regressor, reg_param_grid)

        self._hat_m = regressor_y.fit(x, y.squeeze())


    @fitted
    def score(self, t, m, x, y, hat_tau):
        """Predicts score on data samples

        Parameters
        ----------

        hat_tau array-like, shape (n_samples)
                estimated risk
        """

        hat_e = self._hat_e.predict_proba(x)[:, 1]
        hat_m = self._hat_m.predict(x)
        score = r_risk(y.squeeze(), t.squeeze(), hat_m, hat_e, hat_tau)
        return score
    

    def fit_treatment_propensity_x_nuisance(self, t, x):
        """ Fits the nuisance parameter for the propensity P(T=1|X)
        """
        self._classifier_t_x = self.classifier.fit(x, t)


    def fit_treatment_propensity_xm_nuisance(self, t, m, x):
        """ Fits the nuisance parameter for the propensity P(T=1|X, M)
        """
        xm = np.hstack((x, m))
        self._classifier_t_xm = self.classifier.fit(xm, t)


    def fit_mediator_nuisance(self, t, m, x):
        """ Fits the nuisance parameter for the density f(M=m|T, X)
        """
        # estimate mediator densities
        clf_param_grid = {}
        classifier_m = GridSearchCV(self.classifier, clf_param_grid)

        t_x = _get_interactions(False, t, x)

        # Fit classifier
        self._classifier_m = classifier_m.fit(t_x, m.ravel())


    def fit_conditional_mean_outcome_nuisance(self, t, m, x, y):
        """ Fits the nuisance for the conditional mean outcome for the density f(M=m|T, X)
        """
        if len(m.shape) == 1:
            mr = m.reshape(-1, 1)
        else:
            mr = np.copy(m)
        x_t_mr = _get_interactions(False, x, t, mr)

        reg_param_grid = {}

        # estimate conditional mean outcomes
        regressor_y = GridSearchCV(self.regressor, reg_param_grid)

        self._regressor_y = regressor_y.fit(x_t_mr, y)


    def fit_cross_conditional_mean_outcome_nuisance(self, t, m, x, y):
        """ Fits the cross conditional mean outcome E[E[Y|T=t,M,X]|T=t',X]
        """

        xm = np.hstack((x, m))

        reg_param_grid = {}

        # estimate conditional mean outcomes
        regressor_y = GridSearchCV(self.regressor, reg_param_grid)

        n = t.shape[0]
        train = np.arange(n)
        (
            mu_1mx_nested,  # E[Y|T=1,M,X] predicted on train_nested set
            mu_0mx_nested,  # E[Y|T=0,M,X] predicted on train_nested set
        ) = [np.zeros(n) for _ in range(2)]

        train1 = train[t[train] == 1]
        train0 = train[t[train] == 0]

        train_mean, train_nested = np.array_split(train, 2)
        # train_mean = train
        # train_nested = train
        train_mean1 = train_mean[t[train_mean] == 1]
        train_mean0 = train_mean[t[train_mean] == 0]
        train_nested1 = train_nested[t[train_nested] == 1]
        train_nested0 = train_nested[t[train_nested] == 0]

        self.regressors = {}

        # predict E[Y|T=1,M,X]
        self.regressors['y_t1_mx'] = clone(regressor_y)
        self.regressors['y_t1_mx'].fit(xm[train_mean1], y[train_mean1])
        mu_1mx_nested[train_nested] = self.regressors['y_t1_mx'].predict(xm[train_nested])

        # predict E[Y|T=0,M,X]
        self.regressors['y_t0_mx'] = clone(regressor_y)
        self.regressors['y_t0_mx'].fit(xm[train_mean0], y[train_mean0])
        mu_0mx_nested[train_nested] = self.regressors['y_t0_mx'].predict(xm[train_nested])

        # predict E[E[Y|T=1,M,X]|T=0,X]
        self.regressors['y_t1_x_t0'] = clone(regressor_y)
        self.regressors['y_t1_x_t0'].fit(x[train_nested0], mu_1mx_nested[train_nested0])

        # predict E[E[Y|T=0,M,X]|T=1,X]
        self.regressors['y_t0_x_t1'] = clone(regressor_y)
        self.regressors['y_t0_x_t1'].fit(x[train_nested1], mu_0mx_nested[train_nested1])

        # predict E[Y|T=1,X]
        self.regressors['y_t1_x'] = clone(regressor_y)
        self.regressors['y_t1_x'].fit(x[train1], y[train1])

        # predict E[Y|T=0,X]
        self.regressors['y_t0_x'] = clone(regressor_y)
        self.regressors['y_t0_x'].fit(x[train0], y[train0])


    def fit_cross_conditional_mean_outcome_nuisance_discrete(self, t, m, x, y):
        n = len(y)

        # Initialisation
        (
            mu_1mx,  # E[Y|T=1,M,X]
            mu_0mx,  # E[Y|T=0,M,X]
        ) = [np.zeros(n) for _ in range(2)]

        t0, m0 = np.zeros((n, 1)), np.zeros((n, 1))
        t1, m1 = np.ones((n, 1)), np.ones((n, 1))

        x_t_m = _get_interactions(False, x, t, m)
        x_t1_m = _get_interactions(False, x, t1, m)
        x_t0_m = _get_interactions(False, x, t0, m)

        test_index = np.arange(n)
        ind_t0 = t[test_index] == 0

        reg_param_grid = {}

        # estimate conditional mean outcomes
        regressor_y = GridSearchCV(self.regressor, reg_param_grid)

        self.regressors = {}

        # mu_tm model fitting
        self.regressors['y_t_mx'] = clone(regressor_y).fit(x_t_m, y)

        # predict E[Y|T=t,M,X]
        mu_1mx[test_index] = self.regressors['y_t_mx'].predict(x_t1_m[test_index, :])
        mu_0mx[test_index] = self.regressors['y_t_mx'].predict(x_t0_m[test_index, :])

        for i, b in enumerate(np.unique(m)):
            mb = m1 * b

            mu_1bx, mu_0bx = [np.zeros(n) for h in range(2)]

            # predict E[Y|T=t,M=m,X]
            mu_0bx[test_index] = self.regressors['y_t_mx'].predict(
                _get_interactions(False, x, t0, mb)[test_index, :])
            mu_1bx[test_index] = self.regressors['y_t_mx'].predict(
                _get_interactions(False, x, t1, mb)[test_index, :])

            # E[E[Y|T=1,M=m,X]|T=t,X] model fitting
            self.regressors['reg_y_t1m{}_t0'.format(i)] = clone(
                regressor_y).fit(
                x[test_index, :][ind_t0, :],
                mu_1bx[test_index][ind_t0])
            self.regressors['reg_y_t1m{}_t1'.format(i)] = clone(
                regressor_y).fit(
                x[test_index, :][~ind_t0, :], mu_1bx[test_index][~ind_t0])

            # E[E[Y|T=0,M=m,X]|T=t,X] model fitting
            self.regressors['reg_y_t0m{}_t0'.format(i)] = clone(
                regressor_y).fit(
                x[test_index, :][ind_t0, :],
                mu_0bx[test_index][ind_t0])
            self.regressors['reg_y_t0m{}_t1'.format(i)] = clone(
                regressor_y).fit(
                x[test_index, :][~ind_t0, :],
                mu_0bx[test_index][~ind_t0])

