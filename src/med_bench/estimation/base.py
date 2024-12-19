from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn import clone
from sklearn.model_selection import GridSearchCV

from med_bench.utils.decorators import fitted


class Estimator:
    """General abstract class for causal mediation Estimator"""

    __metaclass__ = ABCMeta

    def __init__(self, verbose: bool = True, crossfit: int = 0):
        """Initializes Estimator base class

        Parameters
        ----------
        verbose : bool
            will print some logs if True
        crossfit : int
            number of crossfit folds, if 0 no crossfit is performed
        """
        self._crossfit = crossfit
        self._crossfit_check()

        self._verbose = verbose

        self._fitted = False

    @property
    def verbose(self):
        return self._verbose

    def _crossfit_check(self):
        """Checks if the estimator inputs are valid"""
        if self._crossfit > 0:
            raise NotImplementedError(
                """Crossfit is not implemented yet
                                      You should perform something like this on your side : 
                                        cf_iterator = KFold(k=5)
                                        for data_train, data_test in cf_iterator:
                                            result.append(DML(...., cross_fitting=False)
                                                .fit(train_data.X, train_data.t, train_data.m, train_data.y)\
                                                .estimate(test_data.X, test_data.t, test_data.m, test_data.y))
                                        np.mean(result)"""
            )

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
            raise ValueError("Inputs don't have the same number of observations")

        y = y.ravel()
        t = t.ravel()

        return t, m, x, y

    def _fit_treatment_propensity_x(self, t, x):
        """Fits the nuisance parameter for the propensity P(T=1|X)"""
        self._classifier_t_x = clone(self.classifier).fit(x, t)

        return self

    def _fit_treatment_propensity_xm(self, t, m, x):
        """Fits the nuisance parameter for the propensity P(T=1|X, M)"""
        xm = np.hstack((x, m))
        self._classifier_t_xm = clone(self.classifier).fit(xm, t)

        return self

    # TODO : Enable any sklearn object as classifier or regressor
    def _fit_binary_mediator_probability(self, t, m, x):
        """Fits the nuisance parameter for the density f(M=m|T, X)"""
        # estimate mediator densities
        t_x = np.hstack([t.reshape(-1, 1), x])

        # Fit classifier
        self._classifier_m = clone(self.classifier).fit(t_x, m.ravel())

        return self

    def _fit_conditional_mean_outcome(self, t, m, x, y):
        """Fits the nuisance for the conditional mean outcome for the density f(M=m|T, X)"""
        x_t_m = np.hstack([x, t.reshape(-1, 1), m])
        self._regressor_y = clone(self.regressor).fit(x_t_m, y)

        return self

    def _fit_cross_conditional_mean_outcome(self, t, m, x, y):
        """Fits the cross conditional mean outcome E[E[Y|T=t,M,X]|T=t',X]"""

        xm = np.hstack((x, m))

        n = t.shape[0]
        train = np.arange(n)
        (
            mu_1mx_nested,  # E[Y|T=1,M,X] predicted on train_nested set
            mu_0mx_nested,  # E[Y|T=0,M,X] predicted on train_nested set
        ) = [np.zeros(n) for _ in range(2)]

        train1 = train[t[train] == 1]
        train0 = train[t[train] == 0]

        train_mean, train_nested = np.array_split(train, 2)
        # train_nested = train
        train_mean1 = train_mean[t[train_mean] == 1]
        train_mean0 = train_mean[t[train_mean] == 0]
        train_nested1 = train_nested[t[train_nested] == 1]
        train_nested0 = train_nested[t[train_nested] == 0]

        self.regressors = {}

        # predict E[Y|T=1,M,X]
        self.regressors["y_t1_mx"] = clone(self.regressor)
        self.regressors["y_t1_mx"].fit(xm[train_mean1], y[train_mean1])
        mu_1mx_nested[train_nested] = self.regressors["y_t1_mx"].predict(
            xm[train_nested]
        )

        # predict E[Y|T=0,M,X]
        self.regressors["y_t0_mx"] = clone(self.regressor)
        self.regressors["y_t0_mx"].fit(xm[train_mean0], y[train_mean0])
        mu_0mx_nested[train_nested] = self.regressors["y_t0_mx"].predict(
            xm[train_nested]
        )

        # predict E[E[Y|T=1,M,X]|T=0,X]
        self.regressors["y_t1_x_t0"] = clone(self.regressor)
        self.regressors["y_t1_x_t0"].fit(x[train_nested0], mu_1mx_nested[train_nested0])

        # predict E[E[Y|T=0,M,X]|T=1,X]
        self.regressors["y_t0_x_t1"] = clone(self.regressor)
        self.regressors["y_t0_x_t1"].fit(x[train_nested1], mu_0mx_nested[train_nested1])

        # predict E[Y|T=1,X]
        self.regressors["y_t1_x"] = clone(self.regressor)
        self.regressors["y_t1_x"].fit(x[train1], y[train1])

        # predict E[Y|T=0,X]
        self.regressors["y_t0_x"] = clone(self.regressor)
        self.regressors["y_t0_x"].fit(x[train0], y[train0])

        return self

    def _estimate_binary_mediator_probability(self, x, m):
        """
        Estimate mediator density P(M=m|T,X) for a binary M

        Returns
        -------
        f_m0x, array-like, shape (n_samples)
            probabilities f(M|T=0,X)
        f_m1x, array-like, shape (n_samples)
            probabilities f(M|T=1,X)
        """
        n = x.shape[0]

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        m = m.ravel()

        t0_x = np.hstack([t0.reshape(-1, 1), x])
        t1_x = np.hstack([t1.reshape(-1, 1), x])

        f_m0x = self._classifier_m.predict_proba(t0_x)[np.arange(m.shape[0]), m]
        f_m1x = self._classifier_m.predict_proba(t1_x)[np.arange(m.shape[0]), m]

        return f_m0x, f_m1x

    def _estimate_binary_mediator_probability_table(self, x):
        """
        Estimate mediator density f(M|T,X)

        Returns
        -------
        f_00x: array-like, shape (n_samples)
            probabilities f(M=0|T=0,X)
        f_01x, array-like, shape (n_samples)
            probabilities f(M=0|T=1,X)
        f_10x, array-like, shape (n_samples)
            probabilities f(M=1|T=0,X)
        f_11x, array-like, shape (n_samples)
            probabilities f(M=1|T=1,X)
        """
        n = x.shape[0]

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        t0_x = np.hstack([t0.reshape(-1, 1), x])
        t1_x = np.hstack([t1.reshape(-1, 1), x])

        # predict f(M=m|T=t,X)
        fm_0 = self._classifier_m.predict_proba(t0_x)
        f_00x = fm_0[:, 0]
        f_01x = fm_0[:, 1]
        fm_1 = self._classifier_m.predict_proba(t1_x)
        f_10x = fm_1[:, 0]
        f_11x = fm_1[:, 1]

        return f_00x, f_01x, f_10x, f_11x

    def _estimate_treatment_propensity_x(self, x):
        """
        Estimate treatment propensity P(T=1|X)

        Returns
        -------
        p_x : array-like, shape (n_samples)
            probabilities P(T=1|X)
        """
        p_x = self._classifier_t_x.predict_proba(x)[:, 1]

        return p_x

    def _estimate_treatment_propensity_xm(self, m, x):
        """
        Estimate treatment probabilities P(T=1|X) and P(T=1|X, M) with train

        Returns
        -------
        p_x : array-like, shape (n_samples)
            probabilities P(T=1|X)
        p_xm : array-like, shape (n_samples)
            probabilities P(T=1|X, M)
        """
        xm = np.hstack((x, m))

        p_xm = self._classifier_t_xm.predict_proba(xm)[:, 1]

        return p_xm

    def _estimate_cross_conditional_mean_outcome(self, m, x):
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
        mu_1mx = self.regressors["y_t1_mx"].predict(xm)

        # predict E[Y|T=0,M,X]
        mu_0mx = self.regressors["y_t0_mx"].predict(xm)

        # predict E[E[Y|T=1,M,X]|T=0,X]
        E_mu_t1_t0 = self.regressors["y_t1_x_t0"].predict(x)

        # predict E[E[Y|T=0,M,X]|T=1,X]
        E_mu_t0_t1 = self.regressors["y_t0_x_t1"].predict(x)

        # predict E[Y|T=1,X]
        mu_1x = self.regressors["y_t1_x"].predict(x)

        # predict E[Y|T=0,X]
        mu_0x = self.regressors["y_t0_x"].predict(x)

        return mu_0mx, mu_1mx, mu_0x, E_mu_t0_t1, E_mu_t1_t0, mu_1x
