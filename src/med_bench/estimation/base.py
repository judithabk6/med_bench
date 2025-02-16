from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn import clone
from sklearn.model_selection import KFold
from med_bench.utils.utils import is_array_integer
from med_bench.utils.decorators import fitted
from med_bench.utils.density import GaussianDensityEstimation
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


class Estimator:
    """General abstract class for causal mediation Estimator"""

    __metaclass__ = ABCMeta

    def __init__(self, verbose: bool = True, mediator_cardinality_threshold: int=10):
        """Initializes Estimator base class

        Parameters
        ----------
        verbose : bool
            will print some logs if True

        mediator_cardinality_threshold: int
            default 10
            maximal number of categories in a mediator to treat it as discrete or continuous
            if the mediator is 1-dimensional and 
                if the number of distinct values in the mediator is lower than
                mediator_cardinality_threshold, the mediator is going to be considered
                discrete when estimating the mediator probability function, otherwise
                the mediator is considered continuous, 
        """
        self._verbose = verbose
        self._fitted = False
        self.mediator_cardinality_threshold = mediator_cardinality_threshold

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
    def _pointwise_estimate(self, t, m, x, y):
        """Point wise estimate of causal effect on data

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

        Returns
        -------
        f_m0x, array-like, shape (n_samples)
            probabilities f(M|T=0,X)
        f_m1x, array-like, shape (n_samples)
            probabilities f(M|T=1,X)
        """
        pass

    @fitted
    def estimate(self, t, m, x, y):
        """Estimate causal effect on data

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

        y0m0, y0m1, y1m0, y1m1 = self._pointwise_estimate(t, m, x, y)

        # effects computing
        total = np.mean(y1m1 - y0m0)
        direct1 = np.mean(y1m1 - y0m1)
        direct0 = np.mean(y1m0 - y0m0)
        indirect1 = np.mean(y1m1 - y1m0)
        indirect0 = np.mean(y0m1 - y0m0)

        causal_effects = {
            "total_effect": total,
            "direct_effect_treated": direct1,
            "direct_effect_control": direct0,
            "indirect_effect_treated": indirect1,
            "indirect_effect_control": indirect0,
        }
        return causal_effects

    def cross_fit_estimate(self, t, m, x, y, n_splits=1):
        """Estimate causal effect on data with cross-fitting

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

        # Initialize KFold for sample splitting
        kfold = KFold(n_splits=n_splits)

        n = t.shape[0]

        # Create placeholders for cross-fitted predictions
        y0m0 = np.zeros(n)
        y0m1 = np.zeros(n)
        y1m0 = np.zeros(n)
        y1m1 = np.zeros(n)

        # Cross-Fitting
        for train_idx, test_idx in kfold.split(x):

            # Train nuisance models on one split
            self.fit(t[train_idx], m[train_idx], x[train_idx], y[train_idx])

            # Predict for the other split
            y0m0_fold, y0m1_fold, y1m0_fold, y1m1_fold = self._pointwise_estimate(
                t[test_idx], m[test_idx], x[test_idx], y[test_idx]
            )

            y0m0[test_idx] = y0m0_fold
            y0m1[test_idx] = y0m1_fold
            y1m0[test_idx] = y1m0_fold
            y1m1[test_idx] = y1m1_fold

        # effects computing
        total = np.mean(y1m1 - y0m0)
        direct1 = np.mean(y1m1 - y0m1)
        direct0 = np.mean(y1m0 - y0m0)
        indirect1 = np.mean(y1m1 - y1m0)
        indirect0 = np.mean(y0m1 - y0m0)

        causal_effects = {
            "total_effect": total,
            "direct_effect_treated": direct1,
            "direct_effect_control": direct0,
            "indirect_effect_treated": indirect1,
            "indirect_effect_control": indirect0,
        }
        return causal_effects

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

    def _fit_mediator_discretizer(self, m):
        """Fits the discretization procedure of mediators"""
        if (is_array_integer(m)) and (len(np.unique(m)) <= self.mediator_cardinality_threshold):
            self.discretizer = LabelEncoder()
            self.discretizer.fit(m.ravel())
            self.mediator_bins = self.discretizer.classes_
            self._mediator_considered_discrete = True
        else:
            self.discretizer = KMeans(n_clusters=10, random_state=42, n_init="auto")
            self.discretizer.fit(m)
            self.mediator_bins = self.discretizer.cluster_centers_
            self._mediator_considered_discrete = False

    def _fit_treatment_propensity_x(self, t, x):
        """Fits the nuisance parameter for the propensity P(T=1|X)"""
        self._classifier_t_x = clone(self.classifier).fit(x, t)

        return self

    def _fit_treatment_propensity_xm(self, t, m, x):
        """Fits the nuisance parameter for the propensity P(T=1|X, M)"""
        xm = np.hstack((x, m))
        self._classifier_t_xm = clone(self.classifier).fit(xm, t)

        return self

    def _fit_mediator_probability(self, t, m, x):
        self._fit_mediator_discretizer(m)
        if not self._mediator_considered_discrete:
            self._fit_mediator_density(t, m, x)
        else:
            m_label, m_discrete_value = self._discretize_mediators(m)
            self._fit_discrete_mediator_probability(t, m_label, x)

    def _fit_discrete_mediator_probability(self, t, m, x):
        """Fits the nuisance parameter for the density f(M=m|T, X)"""
        # estimate mediator densities
        t_x = np.hstack([t.reshape(-1, 1), x])
        # Fit classifier
        self._classifier_m = clone(self.classifier).fit(t_x, m.ravel())

        return self

    def _fit_mediator_density(self, t, m, x):
        """Fits the nuisance parameter for the density f(M=m|T, X)"""
        # estimate mediator densities
        t_x = np.hstack([t.reshape(-1, 1), x])

        self._density_m = GaussianDensityEstimation()
        self._density_m.fit(t_x, m.squeeze())

        return self

    def _fit_conditional_mean_outcome(self, t, m, x, y):
        """Fits the nuisance for the conditional mean outcome for the density f(M=m|T, X)"""
        x_t_m = np.hstack([x, t.reshape(-1, 1), m])
        self._regressor_y = clone(self.regressor).fit(x_t_m, y)

        return self

    def _fit_cross_conditional_mean_outcome(self, t, m, x, y):
        """Fits the cross conditional mean outcome E[E[Y|T=t,M,X]|T=t',X]
        Implicit integration
        """

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

    def _estimate_discrete_mediator_probability(self, x, m_label):
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


        t0_x = np.hstack([t0.reshape(-1, 1), x])
        t1_x = np.hstack([t1.reshape(-1, 1), x])

        f_m0x = self._classifier_m.predict_proba(t0_x)[np.arange(m_label.shape[0]), m_label]
        f_m1x = self._classifier_m.predict_proba(t1_x)[np.arange(m_label.shape[0]), m_label]

        return f_m0x, f_m1x

    def _estimate_mediator_density(self, x, m):
        """
        Estimate mediator density P(M=m|T,X) for a continuous M

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

        m = m.squeeze()

        t0_x = np.hstack([t0.reshape(-1, 1), x])
        t1_x = np.hstack([t1.reshape(-1, 1), x])

        f_m0x = self._density_m.pdf(t0_x, m)
        f_m1x = self._density_m.pdf(t1_x, m)

        return f_m0x, f_m1x

    def _estimate_mediator_probability(self, x, m):
        if not self._mediator_considered_discrete:
            return self._estimate_mediator_density(x, m)
        else:
            m_label, m_discrete_value = self._discretize_mediators(m)
            return self._estimate_discrete_mediator_probability(x, m_label)

    def _estimate_discrete_mediator_probability_table(self, x):
        """
        Estimate mediator discrete probability f(M|T,X)

        Returns
        -------
        f_0x: list, list of array-like of shape (n_samples)
            probabilities f(M=m|T=0,X) for all mediators m
        f_1x, list, list of array-like of shape (n_samples)
            probabilities f(M=m|T=1,X) for all mediators m
        """
        n = x.shape[0]

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        t0_x = np.hstack([t0.reshape(-1, 1), x])
        t1_x = np.hstack([t1.reshape(-1, 1), x])
        f_0x = []
        f_1x = []

        # predict f(M=m|T=t,X)
        fm_0 = self._classifier_m.predict_proba(t0_x)
        fm_1 = self._classifier_m.predict_proba(t1_x)

        for idx, m_anchor in enumerate(self.mediator_bins):
            f_0x.append(fm_0[:, idx])
            f_1x.append(fm_1[:, idx])

        return f_0x, f_1x

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

    def _estimate_conditional_mean_outcome_table(self, x):
        """
        Estimate conditional mean outcome E[Y|T,M,X] for all mediators

        Returns
        -------
        mu_0x: list, list of array-like of shape (n_samples)
            conditional mean outcome estimates E[Y|T=0,M=m,X]
            for all mediators m
        mu_1x, list, list of array-like of shape (n_samples)
            conditional mean outcome estimates E[Y|T=1,M=m,X]
            for all mediators m
        """
        n = x.shape[0]

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))
        mu_0x = list()
        mu_1x = list()

        for m in self.mediator_bins:
            m = m * np.ones((n, 1))
            x_t1_m = np.hstack([x, t1.reshape(-1, 1), m])
            x_t0_m = np.hstack([x, t0.reshape(-1, 1), m])

            mu_0x.append(self._regressor_y.predict(x_t0_m))
            mu_1x.append(self._regressor_y.predict(x_t1_m))

        return mu_0x, mu_1x

    def _estimate_conditional_mean_outcome(self, x, m):
        """
        Estimate conditional mean outcome E[Y|T,M,X]

        Returns
        -------
        mu_0mx: array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=0,M=m,X]
        mu_1mx, array-like, shape (n_samples)
            conditional mean outcome estimates E[Y|T=1,M=m,X]
        """
        n = x.shape[0]

        t0 = np.zeros((n, 1))
        t1 = np.ones((n, 1))

        x_t1_m = np.hstack([x, t1.reshape(-1, 1), m])
        x_t0_m = np.hstack([x, t0.reshape(-1, 1), m])

        mu_1mx = self._regressor_y.predict(x_t1_m)
        mu_0mx = self._regressor_y.predict(x_t0_m)

        return mu_0mx, mu_1mx

    def _estimate_cross_conditional_mean_outcome(self, m, x):
        """
        Estimate the conditional mean outcome,
        the cross conditional mean outcome with an implicit integration

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

    def _discretize_mediators(self, m):
        """Discretize mediators clustering if they are not explicit."""
        if self._mediator_considered_discrete:
            m_label = self.discretizer.transform(m)
            m_discrete_value = m
        else:
            m_label = self.discretizer.predict(m)
            m_discrete_value = self.discretizer.cluster_centers_[m_label, :]
        return m_label, m_discrete_value
