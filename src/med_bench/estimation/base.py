from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone, RegressorMixin, ClassifierMixin

from med_bench.utils.decorators import fitted
from med_bench.utils.scores import r_risk
from med_bench.utils.utils import _get_interactions, _get_train_test_lists


class Estimator:
    """General abstract class for causal mediation Estimator
    """
    __metaclass__ = ABCMeta

    def __init__(self, mediator_type: str, regressor: RegressorMixin, classifier: ClassifierMixin,
                 verbose: bool = True):
        """Initialize Estimator base class

        Parameters
        ----------
        mediator_type : str
            mediator type (binary or continuous, continuous only can be multidimensional)
        regressor : RegressorMixin
            Scikit-Learn Regressor used for mu estimation
        classifier : ClassifierMixin
            Scikit-Learn Classifier used for propensity estimation
        verbose : bool
            will print some logs if True
        """
        self.rng = np.random.RandomState(123)

        assert mediator_type in [
            'binary', 'continuous'], "mediator_type must be 'binary' or 'continuous'"
        self.mediator_type = mediator_type

        self.regressor = regressor

        self.classifier = classifier

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

    def _fit_nuisance(self, t, m, x, y, *args, **kwargs):
        """ Fits the score of the nuisance parameters
        """
        # How do we want to specify gridsearch parameters ? As a function param, a constant or hardcoded here ?
        clf_param_grid = {}
        reg_param_grid = {}

        classifier_x = GridSearchCV(self.classifier, clf_param_grid)

        self._hat_e = classifier_x.fit(x, t.squeeze())

        regressor_y = GridSearchCV(self.regressor, reg_param_grid)

        self._hat_m = regressor_y.fit(x, y.squeeze())

        return self

    @fitted
    def score(self, t, m, x, y, tau_):
        """Predicts score on data samples

        Parameters
        ----------

        tau_ array-like, shape (n_samples)
                estimated risk
        """

        hat_e = self._hat_e.predict_proba(x)[:, 1]
        hat_m = self._hat_m.predict(x)
        score = r_risk(y.squeeze(), t.squeeze(), hat_m, hat_e, tau_)
        return score

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

    def _fit_mediator_nuisance(self, t, m, x):
        """ Fits the nuisance parameter for the density f(M=m|T, X)
        """
        # estimate mediator densities
        clf_param_grid = {}
        classifier_m = GridSearchCV(self.classifier, clf_param_grid)

        t_x = _get_interactions(False, t, x)

        # Fit classifier
        self._classifier_m = classifier_m.fit(t_x, m.ravel())

        return self

    def _fit_conditional_mean_outcome_nuisance(self, t, m, x, y):
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
        mu_1mx[test_index] = self.regressors['y_t_mx'].predict(
            x_t1_m[test_index, :])
        mu_0mx[test_index] = self.regressors['y_t_mx'].predict(
            x_t0_m[test_index, :])

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
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if len(t.shape) == 1:
            t = t.reshape(-1, 1)

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        m = m.ravel()

        train_test_list = _get_train_test_lists(self._crossfit, n, x)

        f_m0x, f_m1x = [np.zeros(n) for h in range(2)]

        t_x = _get_interactions(False, t, x)
        t0_x = _get_interactions(False, t0, x)
        t1_x = _get_interactions(False, t1, x)

        for _, test_index in train_test_list:

            test_ind = np.arange(len(test_index))

            fm_0 = self._classifier_m.predict_proba(t0_x[test_index, :])
            fm_1 = self._classifier_m.predict_proba(t1_x[test_index, :])

            # predict f(M|T=t,X)
            f_m0x[test_index] = fm_0[test_ind, m[test_index]]
            f_m1x[test_index] = fm_1[test_ind, m[test_index]]

            for i, b in enumerate(np.unique(m)):
                f_0bx, f_1bx = [np.zeros(n) for h in range(2)]

                # predict f(M=m|T=t,X)
                f_0bx[test_index] = fm_0[:, i]
                f_1bx[test_index] = fm_1[:, i]

        return f_m0x, f_m1x

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
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if len(t.shape) == 1:
            t = t.reshape(-1, 1)

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        m = m.ravel()

        train_test_list = _get_train_test_lists(self._crossfit, n, x)

        f_t1, f_t0 = [], []

        t_x = _get_interactions(False, t, x)
        t0_x = _get_interactions(False, t0, x)
        t1_x = _get_interactions(False, t1, x)

        for _, test_index in train_test_list:

            fm_0 = self._classifier_m.predict_proba(t0_x[test_index, :])
            fm_1 = self._classifier_m.predict_proba(t1_x[test_index, :])

            for i, b in enumerate(np.unique(m)):
                f_0bx, f_1bx = [np.zeros(n) for h in range(2)]

                # predict f(M=m|T=t,X)
                f_0bx[test_index] = fm_0[:, i]
                f_1bx[test_index] = fm_1[:, i]

                f_t0.append(f_0bx)
                f_t1.append(f_1bx)

        return f_t0, f_t1

    def _estimate_treatment_propensity_x(self, t, m, x):
        """
        Estimate treatment probabilities P(T=1|X) with train
        test lists from crossfitting

        Returns
        -------
        p_x : array-like, shape (n_samples)
            probabilities P(T=1|X)
        p_xm : array-like, shape (n_samples)
            probabilities P(T=1|X, M)
        """
        n = len(t)

        p_x, p_xm = [np.zeros(n) for h in range(2)]
        # compute propensity scores
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(m.shape) == 1:
            m = m.reshape(-1, 1)
        if len(t.shape) == 1:
            t = t.reshape(-1, 1)

        train_test_list = _get_train_test_lists(self._crossfit, n, x)

        for _, test_index in train_test_list:

            # predict P(T=1|X), P(T=1|X, M)
            p_x[test_index] = self._classifier_t_x.predict_proba(x[test_index, :])[
                :, 1]

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
        n = len(t)

        p_x, p_xm = [np.zeros(n) for h in range(2)]
        # compute propensity scores
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(m.shape) == 1:
            m = m.reshape(-1, 1)
        if len(t.shape) == 1:
            t = t.reshape(-1, 1)

        train_test_list = _get_train_test_lists(self._crossfit, n, x)

        xm = np.hstack((x, m))

        for _, test_index in train_test_list:

            # predict P(T=1|X), P(T=1|X, M)
            p_x[test_index] = self._classifier_t_x.predict_proba(x[test_index, :])[
                :, 1]
            p_xm[test_index] = self._classifier_t_xm.predict_proba(xm[test_index, :])[
                :, 1]

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
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(m.shape) == 1:
            mr = m.reshape(-1, 1)
        else:
            mr = np.copy(m)
        if len(t.shape) == 1:
            t = t.reshape(-1, 1)

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))
        m1 = np.ones((n, 1))

        train_test_list = _get_train_test_lists(self._crossfit, n, x)

        mu_1mx, mu_0mx = [np.zeros(n) for _ in range(2)]
        mu_t1, mu_t0 = [], []

        m1 = np.ones((n, 1))

        x_t_mr = _get_interactions(False, x, t, mr)
        x_t1_m = _get_interactions(False, x, t1, m)
        x_t0_m = _get_interactions(False, x, t0, m)

        for _, test_index in train_test_list:

            # predict E[Y|T=t,M,X]
            mu_0mx[test_index] = self._regressor_y.predict(
                x_t0_m[test_index, :]).squeeze()
            mu_1mx[test_index] = self._regressor_y.predict(
                x_t1_m[test_index, :]).squeeze()

            for i, b in enumerate(np.unique(m)):
                mu_1bx, mu_0bx = [np.zeros(n) for h in range(2)]
                mb = m1 * b

                # predict E[Y|T=t,M=m,X]
                mu_0bx[test_index] = self._regressor_y.predict(
                    _get_interactions(False, x, t0, mb)[test_index,
                                                        :]).squeeze()
                mu_1bx[test_index] = self._regressor_y.predict(
                    _get_interactions(False, x, t1, mb)[test_index,
                                                        :]).squeeze()

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
        n = len(y)

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
