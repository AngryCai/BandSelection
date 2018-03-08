"""
Ref:
    Zheng Zhao and Huan Liu. 2007. Spectral feature selection for supervised and unsupervised learning. In Proceedings
    of the 24th international conference on Machine learning (ICML '07), Zoubin Ghahramani (Ed.). ACM, New York, NY,
    USA, 1151-1157.
"""
from skfeature.function.similarity_based import SPEC
import numpy as np


class SPEC_HSI(object):

    def __init__(self, n_band=10):
        self.n_band = n_band

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        """

        :param X: shape [n_row*n_clm, n_band]
        :return:
        """
        # specify the second ranking function which uses all except the 1st eigenvalue
        kwargs = {'style': 0}
        # n_row, n_column, __n_band = X.shape
        # XX = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        XX = X

        # obtain the scores of features
        score = SPEC.spec(XX, **kwargs)

        # sort the feature scores in an descending order according to the feature scores
        idx = SPEC.feature_ranking(score, **kwargs)

        # obtain the dataset on the selected features
        selected_features = XX[:, idx[0:self.n_band]]
        # selected_features.reshape((self.n_band, n_row, n_column))
        # selected_features = np.transpose(selected_features, axes=(1, 2, 0))
        return selected_features
