"""
Nonnegative Discriminative Feature Selection (NDFS)
Reference:
    Li, Zechao, et al. "Unsupervised Feature Selection Using Nonnegative Spectral Analysis." AAAI. 2012.
"""
import numpy as np
from skfeature.function.sparse_learning_based import NDFS
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking


class NDFS_HSI(object):

    def __init__(self, n_cluster, n_band=10):
        self.n_band = n_band
        self.n_cluster = n_cluster

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X):
        """

        :param X: shape [n_row*n_clm, n_band]
        :return:
        """
        # construct affinity matrix
        kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 1}
        W = construct_W.construct_W(X, **kwargs)

        # obtain the feature weight matrix
        Weight = NDFS.ndfs(X, W=W, n_clusters=self.n_cluster)

        # sort the feature scores in an ascending order according to the feature scores
        idx = feature_ranking(Weight)

        # obtain the dataset on the selected features
        selected_features = X[:, idx[0:self.n_band]]
        return selected_features
