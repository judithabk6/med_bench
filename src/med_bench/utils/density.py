import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import itertools
import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp

from sklearn.linear_model import LinearRegression
from scipy.stats import norm, rv_continuous

MULTIPROC_THRESHOLD = 10**4


class GaussianDensityEstimation:
    """Gaussian Density

    Args:
        name: (str) name / identifier of estimator
        bandwidth: scale / bandwith of the gaussian kernels
        n_centers: Number of kernels to use in the output

        random_seed: (optional) seed (int) of the random number generators used
    """

    def __init__(self, name="Gaussian", bandwidth=0.5, random_seed=None):

        self.name = name
        self.random_state = np.random.RandomState(seed=random_seed)
        self.random_seed = random_seed

        self.bandwidth = bandwidth

        self.fitted = False
        self.can_sample = False
        self.reg = LinearRegression()

    def fit(self, X, Y, **kwargs):
        """Fits the conditional density model with provided data

        Args:
          X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
          Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        """
        # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)

        # X, Y = self._handle_input_dimensionality(X, Y, fitting=True)
        self.reg.fit(X, Y)
        self.scale = np.sqrt(np.mean((Y - self.reg.predict(X)) ** 2))
        self.fitted = True

    def representation(self, X):
        return self.reg.predict(X)

    def pdf(self, X, Y):
        """Predicts the conditional density p(y|x). Requires the model to be fitted.

        Args:
          X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
          Y: numpy array of y targets - shape: (n_samples, n_dim_y)

        Returns:
           conditional probability density p(y|x) - numpy array of shape (n_query_samples, )

        """
        assert self.fitted, "model must be fitted for predictions"

        representation = self.representation(X).squeeze()

        pdf_values = norm(loc=representation, scale=self.scale).pdf(Y)
        values_pdf = np.prod(pdf_values, axis=-1) if pdf_values.ndim > 1 else pdf_values

        return values_pdf
