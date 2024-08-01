"""
the objective of this script is to provide nuisance estimators 
for mediation in causal inference
"""

import numpy as np

from sklearn.base import clone

from med_bench.utils.utils import _get_train_test_lists, _get_interactions
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.base import BaseEstimator
from joblib import Parallel, delayed

def estimate_mediator_density(t, m, x, y, crossfit, clf_m,
                                        interaction, fit=False):
    """
    Estimate mediator density f(M|T,X)
    with train test lists from crossfitting

    Returns
    -------
    f_t0: list
        contains array-like, shape (n_samples) probabilities f(M=m|T=0,X)
    f_t1, list
        contains array-like, shape (n_samples) probabilities f(M=m|T=1,X)
    f_m0x, array-like, shape (n_samples)
        probabilities f(M|T=0,X)
    f_m1x, array-like, shape (n_samples)
        probabilities f(M|T=1,X)
    """
    # if not is_array_integer(m):
    #     return estimate_mediator_density_kde(t, m, x, y, crossfit, interaction)
    # else:
    return estimate_mediator_probability(t, m, x, y, crossfit, clf_m,
                                        interaction, fit=False)


def estimate_mediators_probabilities(t, m, x, y, crossfit, clf_m,
                                        interaction, fit=False):
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

    train_test_list = _get_train_test_lists(crossfit, n, x)

    f_t1, f_t0 = [], []

    t_x = _get_interactions(interaction, t, x)
    t0_x = _get_interactions(interaction, t0, x)
    t1_x = _get_interactions(interaction, t1, x)

    for train_index, test_index in train_test_list:


        # f_mtx model fitting
        if fit == True:
            clf_m = clf_m.fit(t_x[train_index, :], m[train_index])

        fm_0 = clf_m.predict_proba(t0_x[test_index, :])
        fm_1 = clf_m.predict_proba(t1_x[test_index, :])


        for i, b in enumerate(np.unique(m)):
            f_0bx, f_1bx = [np.zeros(n) for h in range(2)]

            # predict f(M=m|T=t,X)
            f_0bx[test_index] = fm_0[:, i]
            f_1bx[test_index] = fm_1[:, i]

            f_t0.append(f_0bx)
            f_t1.append(f_1bx)

    return f_t0, f_t1

def estimate_mediator_probability(t, m, x, y, crossfit, clf_m,
                                        interaction, fit=False):
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

    train_test_list = _get_train_test_lists(crossfit, n, x)

    f_m0x, f_m1x = [np.zeros(n) for h in range(2)]

    t_x = _get_interactions(interaction, t, x)
    t0_x = _get_interactions(interaction, t0, x)
    t1_x = _get_interactions(interaction, t1, x)

    for train_index, test_index in train_test_list:

        test_ind = np.arange(len(test_index))

        # f_mtx model fitting
        if fit == True:
            clf_m = clf_m.fit(t_x[train_index, :], m[train_index])

        fm_0 = clf_m.predict_proba(t0_x[test_index, :])
        fm_1 = clf_m.predict_proba(t1_x[test_index, :])

        # predict f(M|T=t,X)
        f_m0x[test_index] = fm_0[test_ind, m[test_index]]
        f_m1x[test_index] = fm_1[test_ind, m[test_index]]

        for i, b in enumerate(np.unique(m)):
            f_0bx, f_1bx = [np.zeros(n) for h in range(2)]

            # predict f(M=m|T=t,X)
            f_0bx[test_index] = fm_0[:, i]
            f_1bx[test_index] = fm_1[:, i]

    return f_m0x, f_m1x

class ConditionalNearestNeighborsKDE(BaseEstimator):
    """Conditional Kernel Density Estimation using nearest neighbors.

    This class implements a Conditional Kernel Density Estimation by applying
    the Kernel Density Estimation algorithm after a nearest neighbors search.

    It allows the use of user-specified nearest neighbor and kernel density
    estimators or, if not provided, defaults will be used.

    Parameters
    ----------
    nn_estimator : NearestNeighbors instance, default=None
        A pre-configured instance of a `~sklearn.neighbors.NearestNeighbors` class
        to use for finding nearest neighbors. If not specified, a
        `~sklearn.neighbors.NearestNeighbors` instance with `n_neighbors=100`
        will be used.

    kde_estimator : KernelDensity instance, default=None
        A pre-configured instance of a `~sklearn.neighbors.KernelDensity` class
        to use for estimating the kernel density. If not specified, a
        `~sklearn.neighbors.KernelDensity` instance with `bandwidth="scott"`
        will be used.
    """

    def __init__(self, nn_estimator=None, kde_estimator=None):
        self.nn_estimator = nn_estimator
        self.kde_estimator = kde_estimator

    def fit(self, X, y=None):
        if self.nn_estimator is None:
            self.nn_estimator_ = NearestNeighbors(n_neighbors=100)
        else:
            self.nn_estimator_ = clone(self.nn_estimator)
        self.nn_estimator_.fit(X, y)
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the conditional density estimation of new samples.

        The predicted density of the target for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be estimated, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        kernel_density_list : list of len n_samples of KernelDensity instances
            Estimated conditional density estimations in the form of
            `~sklearn.neighbors.KernelDensity` instances.
        """
        _, ind_X = self.nn_estimator_.kneighbors(X)
        if self.kde_estimator is None:
            kernel_density_list = [
                KernelDensity(bandwidth="scott").fit(
                    self.y_train_[ind].reshape(-1, 1))
                for ind in ind_X
            ]
        else:
            kernel_density_list = [
                clone(self.kde_estimator).fit(
                    self.y_train_[ind].reshape(-1, 1))
                for ind in ind_X
            ]
        return kernel_density_list

    def pdf(self, y, x):

        ckde_preds = self.predict(x)

        def _evaluate_individual(y_, cde_pred):
            # The score_samples and score methods returns stuff on log scale,
            # so we have to exp it.
            expected_value = np.exp(cde_pred.score(y_.reshape(-1, 1)))
            return expected_value

        individual_predictions = Parallel(n_jobs=-1)(
            delayed(_evaluate_individual)(y_, cde_pred)
            for y_, cde_pred in zip(y, ckde_preds)
        )

        return individual_predictions

# def estimate_mediator_density_kde(t, m, x, y, crossfit, interaction):
#     """
#     Estimate mediator density f(M|T,X)
#     with train test lists from crossfitting

#     Returns
#     -------
#     f_m0x, array-like, shape (n_samples)
#         probabilities f(M|T=0,X)
#     f_m1x, array-like, shape (n_samples)
#         probabilities f(M|T=1,X)
#     """
#     n = len(y)
#     if len(x.shape) == 1:
#         x = x.reshape(-1, 1)

#     if len(t.shape) == 1:
#         t = t.reshape(-1, 1)

#     t0 = np.zeros((n, 1))
#     t1 = np.ones((n, 1))


#     train_test_list = _get_train_test_lists(crossfit, n, x)

#     f_m0x, f_m1x = [np.zeros(n) for _ in range(2)]

#     t_x = _get_interactions(interaction, t, x)
#     t0_x = _get_interactions(interaction, t0, x)
#     t1_x = _get_interactions(interaction, t1, x)

#     for train_index, test_index in train_test_list:

#         # f_mtx model fitting
#         ckde_m = ConditionalNearestNeighborsKDE().fit(t_x[train_index, :],
#                                                       m[train_index, :])

#         # predict f(M|T=t,X)
#         f_m0x[test_index] = ckde_m.pdf(m[test_index, :], t0_x[test_index, :])
#         f_m1x[test_index] = ckde_m.pdf(m[test_index, :], t1_x[test_index, :])

#     return f_m0x, f_m1x

def estimate_mediator_density_kde(t, m, x, y, crossfit, ckde_m, interaction):
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


    f_m0x, f_m1x = [np.zeros(n) for _ in range(2)]

    t_x = _get_interactions(interaction, t, x)
    t0_x = _get_interactions(interaction, t0, x)
    t1_x = _get_interactions(interaction, t1, x)


    # predict f(M|T=t,X)
    f_m0x = ckde_m.pdf(m, t0_x)
    f_m1x = ckde_m.pdf(m, t1_x)

    return f_m0x, f_m1x