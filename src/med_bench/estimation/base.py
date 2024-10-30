from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone, RegressorMixin, ClassifierMixin

from med_bench.utils.decorators import fitted
from med_bench.utils.scores import r_risk
from med_bench.utils.utils import _get_train_test_lists


class Estimator:
    """General abstract class for causal mediation Estimator
    """
    __metaclass__ = ABCMeta

    def __init__(self, regressor, classifier, verbose: bool = True, crossfit: int = 0):
        """Initialize Estimator base class

        Parameters
        ----------
        regressor 
            Regressor used for mu estimation, can be any object with a fit and predict method
        classifier 
            Classifier used for propensity estimation, can be any object with a fit and predict_proba method
        verbose : bool
            will print some logs if True
        crossfit : int
            number of crossfit folds, if 0 no crossfit is performed
        """
        assert hasattr(
            regressor, 'fit'), "The model does not have a 'fit' method."
        assert hasattr(
            regressor, 'predict'), "The model does not have a 'predict' method."
        assert hasattr(
            classifier, 'fit'), "The model does not have a 'fit' method."
        assert hasattr(
            classifier, 'predict_proba'), "The model does not have a 'predict_proba' method."
        self.regressor = regressor
        self.classifier = classifier

        self._crossfit = crossfit
        self._crossfit_check()

        self._verbose = verbose

        self._fitted = False

    @property
    def verbose(self):
        return self._verbose

    def _crossfit_check(self):
        """Checks if the estimator inputs are valid
        """
        if self._crossfit > 0:
            raise NotImplementedError("""Crossfit is not implemented yet
                                      You should perform something like this on your side : 
                                        cf_iterator = KFold(k=5)
                                        for data_train, data_test in cf_iterator:
                                            result.append(DML(...., cross_fitting=False)
                                                .fit(train_data.X, train_data.t, train_data.m, train_data.y)\
                                                .estimate(test_data.X, test_data.t, test_data.m, test_data.y))
                                        np.mean(result)""")

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

    def _resize(self, t, m, x, y):
        """Resize data for the right shape

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
        if len(y) != len(y.ravel()):
            raise ValueError("Multidimensional y is not supported")
        if len(t) != len(t.ravel()):
            raise ValueError("Multidimensional t is not supported")

        n = len(y)
        if len(x.shape) == 1:
            x.reshape(n, 1)
        if len(m.shape) == 1:
            m = m.reshape(n, 1)

        if n != len(x) or n != len(m) or n != len(t):
            raise ValueError(
                "Inputs don't have the same number of observations")

        y = y.ravel()
        t = t.ravel()

        return t, m, x, y

    def _input_reshape(self, t, m, x):
        """Reshape data for the right shape
        """
        if len(t.shape) == 1:
            t = t.reshape(-1, 1)
        if len(m.shape) == 1:
            m = m.reshape(-1, 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        return t, m, x

    def _fit_treatment_propensity_x_nuisance(self, t, x):
        """ Fits the nuisance parameter for the propensity P(T=1|X)
        """
        self._classifier_t_x = self.classifier.fit(x, t)

        return self

    def _fit_treatment_propensity_xm_nuisance(self, t, m, x):
        """ Fits the nuisance parameter for the propensity P(T=1|X, M)
        """
        xm = np.hstack((x, m))
        self._classifier_t_xm = self.classifier.fit(xm, t)

        return self

    # TODO : Enable any sklearn object as classifier or regressor
    def _fit_mediator_nuisance(self, t, m, x):
        """ Fits the nuisance parameter for the density f(M=m|T, X)
        """
        # estimate mediator densities
        clf_param_grid = {}
        classifier_m = GridSearchCV(self.classifier, clf_param_grid)

        t_x = np.hstack([t.reshape(-1, 1), x])

        # Fit classifier
        self._classifier_m = classifier_m.fit(t_x, m.ravel())

        return self

    def _fit_conditional_mean_outcome_nuisance(self, t, m, x, y):
        """ Fits the nuisance for the conditional mean outcome for the density f(M=m|T, X)
        """
        x_t_m = np.hstack([x, t.reshape(-1, 1), m])

        reg_param_grid = {}

        # estimate conditional mean outcomes
        regressor_y = GridSearchCV(self.regressor, reg_param_grid)

        self._regressor_y = regressor_y.fit(x_t_m, y)

        return self

    def _fit_cross_conditional_mean_outcome_nuisance(self, t, m, x, y):
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
        mu_1mx_nested[train_nested] = self.regressors['y_t1_mx'].predict(
            xm[train_nested])

        # predict E[Y|T=0,M,X]
        self.regressors['y_t0_mx'] = clone(regressor_y)
        self.regressors['y_t0_mx'].fit(xm[train_mean0], y[train_mean0])
        mu_0mx_nested[train_nested] = self.regressors['y_t0_mx'].predict(
            xm[train_nested])

        # predict E[E[Y|T=1,M,X]|T=0,X]
        self.regressors['y_t1_x_t0'] = clone(regressor_y)
        self.regressors['y_t1_x_t0'].fit(
            x[train_nested0], mu_1mx_nested[train_nested0])

        # predict E[E[Y|T=0,M,X]|T=1,X]
        self.regressors['y_t0_x_t1'] = clone(regressor_y)
        self.regressors['y_t0_x_t1'].fit(
            x[train_nested1], mu_0mx_nested[train_nested1])

        # predict E[Y|T=1,X]
        self.regressors['y_t1_x'] = clone(regressor_y)
        self.regressors['y_t1_x'].fit(x[train1], y[train1])

        # predict E[Y|T=0,X]
        self.regressors['y_t0_x'] = clone(regressor_y)
        self.regressors['y_t0_x'].fit(x[train0], y[train0])

        return self

    def _fit_cross_conditional_mean_outcome_nuisance_discrete(self, t, m, x, y):
        """
        Fits the cross conditional mean outcome E[E[Y|T=t,M,X]|T=t',X] discrete
        """
        n = len(y)

        # Initialisation
        (
            mu_1mx,  # E[Y|T=1,M,X]
            mu_0mx,  # E[Y|T=0,M,X]
        ) = [np.zeros(n) for _ in range(2)]

        t0, m0 = np.zeros((n, 1)), np.zeros((n, 1))
        t1, m1 = np.ones((n, 1)), np.ones((n, 1))

        x_t_m = np.hstack([x, t.reshape(-1, 1), m])
        x_t1_m = np.hstack([x, t1.reshape(-1, 1), m])
        x_t0_m = np.hstack([x, t0.reshape(-1, 1), m])

        test_index = np.arange(n)
        ind_t0 = t[test_index] == 0

        reg_param_grid = {}

        # estimate conditional mean outcomes
        regressor_y = GridSearchCV(self.regressor, reg_param_grid)

        self.regressors = {}

        # mu_tm model fitting
        self.regressors['y_t_mx'] = clone(regressor_y).fit(x_t_m, y)

        # predict E[Y|T=t,M,X]
        mu_1mx[test_index] = self.regressors['y_t_mx'].predict(
            x_t1_m[test_index, :])
        mu_0mx[test_index] = self.regressors['y_t_mx'].predict(
            x_t0_m[test_index, :])

        for i, b in enumerate(np.unique(m)):
            mb = m1 * b

            mu_1bx, mu_0bx = [np.zeros(n) for h in range(2)]

            # predict E[Y|T=t,M=m,X]

            x_t1_mb = np.hstack([x, t1.reshape(-1, 1), mb])
            x_t0_mb = np.hstack([x, t0.reshape(-1, 1), mb])

            mu_0bx[test_index] = self.regressors['y_t_mx'].predict(
                x_t0_mb[test_index, :])
            mu_1bx[test_index] = self.regressors['y_t_mx'].predict(
                x_t1_mb[test_index, :])

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

        return self

    def _estimate_mediator_probability(self, t, m, x, y):
        """
        Estimate mediator density f(M|T,X)
        with train test lists from crossfitting

        Returns
        -------
        f_m0x, array-like, shape (n_samples)
            probabilities f(M|T=0,X)
        f_m1x, array-like, shape (n_samples)
            probabilities f(M|T=1,X)
        """
        n = len(y)

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        m = m.ravel()

        t0_x = np.hstack([t0.reshape(-1, 1), x])
        t1_x = np.hstack([t1.reshape(-1, 1), x])

        fm_0 = self._classifier_m.predict_proba(t0_x)
        fm_1 = self._classifier_m.predict_proba(t1_x)

        return fm_0, fm_1

    def _estimate_mediators_probabilities(self, t, m, x, y):
        """
        Estimate mediator density f(M|T,X)
        with train test lists from crossfitting

        Returns
        -------
        f_t0: list
            contains array-like, shape (n_samples) probabilities f(M=m|T=0,X)
        f_t1, list
            contains array-like, shape (n_samples) probabilities f(M=m|T=1,X)
        """
        n = len(y)

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        m = m.ravel()

        t0_x = np.hstack([t0.reshape(-1, 1), x])
        t1_x = np.hstack([t1.reshape(-1, 1), x])

        f_t0 = self._classifier_m.predict_proba(t0_x)
        f_t1 = self._classifier_m.predict_proba(t1_x)

        return f_t0, f_t1

    def _estimate_treatment_propensity_x(self, t, m, x):
        """
        Estimate treatment propensity P(T=1|X)

        Returns
        -------
        p_x : array-like, shape (n_samples)
            probabilities P(T=1|X)
        """
        n = len(t)

        # compute propensity scores
        t, m, x = self._input_reshape(t, m, x)

        # predict P(T=1|X), P(T=1|X, M)
        p_x = self._classifier_t_x.predict_proba(x)

        return p_x

    def _estimate_treatment_probabilities(self, t, m, x):
        """
        Estimate treatment probabilities P(T=1|X) and P(T=1|X, M) with train
        test lists from crossfitting

        Returns
        -------
        p_x : array-like, shape (n_samples)
            probabilities P(T=1|X)
        p_xm : array-like, shape (n_samples)
            probabilities P(T=1|X, M)
        """
        # compute propensity scores
        t, m, x = self._input_reshape(t, m, x)

        xm = np.hstack((x, m))

        # predict P(T=1|X), P(T=1|X, M)
        p_x = self._classifier_t_x.predict_proba(x)
        p_xm = self._classifier_t_xm.predict_proba(xm)

        return p_x, p_xm

    def _estimate_conditional_mean_outcome(self, t, m, x, y):
        """
        Estimate conditional mean outcome E[Y|T,M,X]
        with train test lists from crossfitting

        Returns
        -------
        mu_t0: list
            contains array-like, shape (n_samples) conditional mean outcome estimates E[Y|T=0,M=m,X]
        mu_t1, list
            contains array-like, shape (n_samples) conditional mean outcome estimates E[Y|T=1,M=m,X]
        mu_m0x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=0,M,X]
        mu_m1x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=1,M,X]
        """
        n = len(y)

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))
        m1 = np.ones((n, 1))

        mu_t1, mu_t0 = [], []

        m1 = np.ones((n, 1))

        x_t1_m = np.hstack([x, t1.reshape(-1, 1), m])
        x_t0_m = np.hstack([x, t0.reshape(-1, 1), m])

        # predict E[Y|T=t,M,X] for all indices
        mu_0mx = self._regressor_y.predict(x_t0_m).squeeze()
        mu_1mx = self._regressor_y.predict(x_t1_m).squeeze()

        for i, b in enumerate(np.unique(m)):
            mb = m1 * b
            x_t1_mb = np.hstack([x, t1.reshape(-1, 1), mb])
            x_t0_mb = np.hstack([x, t0.reshape(-1, 1), mb])
            # predict E[Y|T=t,M=m,X] for all indices
            mu_0bx = self._regressor_y.predict(x_t0_mb).squeeze()
            mu_1bx = self._regressor_y.predict(x_t1_mb).squeeze()
            mu_t0.append(mu_0bx)
            mu_t1.append(mu_1bx)

        return mu_t0, mu_t1, mu_0mx, mu_1mx

    def _estimate_cross_conditional_mean_outcome_nesting(self, m, x, y):
        """
        Estimate the conditional mean outcome,
        the cross conditional mean outcome

        Returns
        -------
        mu_m0x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=0,M,X]
        mu_m1x, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=1,M,X]
        mu_0x, array-like, shape (n_samples)
            cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=0,X]
        E_mu_t0_t1, array-like, shape (n_samples)
            cross conditional mean outcome estimates E[E[Y|T=0,M,X]|T=1,X]
        E_mu_t1_t0, array-like, shape (n_samples)
            cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=0,X]
        mu_1x, array-like, shape (n_samples)
            cross conditional mean outcome estimates E[E[Y|T=1,M,X]|T=1,X]
        """
        xm = np.hstack((x, m))

        # predict E[Y|T=1,M,X]
        mu_1mx = self.regressors['y_t1_mx'].predict(xm)

        # predict E[Y|T=0,M,X]
        mu_0mx = self.regressors['y_t0_mx'].predict(xm)

        # predict E[E[Y|T=1,M,X]|T=0,X]
        E_mu_t1_t0 = self.regressors['y_t1_x_t0'].predict(x)

        # predict E[E[Y|T=0,M,X]|T=1,X]
        E_mu_t0_t1 = self.regressors['y_t0_x_t1'].predict(x)

        # predict E[Y|T=1,X]
        mu_1x = self.regressors['y_t1_x'].predict(x)

        # predict E[Y|T=0,X]
        mu_0x = self.regressors['y_t0_x'].predict(x)

        return mu_0mx, mu_1mx, mu_0x, E_mu_t0_t1, E_mu_t1_t0, mu_1x
