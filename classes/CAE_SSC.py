"""
An improved Band Selection method using CAE feature.
"""
import numpy as np
from skfeature.function.sparse_learning_based import NDFS
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking
from sklearn.cluster.spectral import SpectralClustering
from sklearn.cluster import KMeans


class CAE_BS(object):
    """
    :argument:
        Implementation of L2 norm based sparse self-expressive clustering model
        with affinity measurement basing on angular similarity
    """
    def __init__(self, n_band=10, coef_=1):
        self.n_band = n_band
        self.coef_ = coef_

    def fit(self, X):
        self.X = X
        return self

    def predict(self, X_cae_fea, X_origin):
        """
        :param X_cae_fea: shape [n_CAE_fea, n_band]
        :param X_origin: original HSI data with a 2-D shape of (n_row*n_clm, n_band)
        :return: selected band subset
        """
        cluster_res = self.__get_cluster_close(X_cae_fea)
        selected_band = self.__get_band(cluster_res, X_origin)
        return selected_band

    def __get_band(self, cluster_result, X):
        """
        select band according to the center of each cluster
        :param cluster_result:
        :param X:
        :return:
        """
        selected_band = []
        n_cluster = np.unique(cluster_result).__len__()
        # img_ = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        for c in np.unique(cluster_result):
            idx = np.nonzero(cluster_result == c)
            center = np.mean(X[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(X[:, idx[0]] - center, axis=0)
            band_ = X[:, idx[0]][:, distance.argmin()]
            selected_band.append(band_)
        bands = np.asarray(selected_band).transpose()
        # bands = bands.reshape(n_cluster, n_row, n_column)
        # bands = np.transpose(bands, axes=(1, 2, 0))
        return bands

    def __get_cluster_close(self, X):
        """
        using close-form solution
        :param X:
        :return:
        """
        n_sample = X.transpose().shape[0]
        H = X.transpose()    # NRP_ELM(self.n_hidden, sparse=False).fit(X).predict(X)
        C = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            y_i = H[i]
            H_i = np.delete(H, i, axis=0).transpose()
            term_1 = np.linalg.inv(np.dot(H_i.transpose(), H_i) + self.coef_ * np.eye(n_sample - 1))
            w = np.dot(np.dot(term_1, H_i.transpose()), y_i.reshape((y_i.shape[0], 1)))
            w = w.flatten()
            #  Normalize the columns of C: ci = ci / ||ci||_ss.
            coef = w / np.max(np.abs(w))
            C[:i, i] = coef[:i]
            C[i + 1:, i] = coef[i:]
        # compute affinity matrix
        L = 0.5 * (np.abs(C) + np.abs(C.T))  # affinity graph
        self.affinity_matrix = L
        # spectral clustering
        sc = SpectralClustering(n_clusters=self.n_band, affinity='precomputed')
        sc.fit(self.affinity_matrix)
        return sc.labels_

