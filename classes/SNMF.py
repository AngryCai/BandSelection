# coding:utf-8

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import nimfa
from sklearn.metrics import accuracy_score

class BandSelection_SNMF(object):
    def __init__(self, n_band):
        self.n_band = n_band

    def predict(self, X):
        """
        :param X: with shape (n_pixel, n_band)
        :return:
        """
        # # Note that X has to reshape to (n_fea., n_sample)
        # XX = X.transpose()  # (n_band, n_pixel)
        # snmf = nimfa.Snmf(X, seed="random_c", rank=self.n_band)  # remain para. default
        snmf = nimfa.Snmf(X, rank=self.n_band, max_iter=20, version='r', eta=1.,
                          beta=1e-4, i_conv=10, w_min_change=0)
        snmf_fit = snmf()
        W = snmf.basis()  # shape: n_band * k
        H = snmf.coef()  # shape: k * n_pixel

        #  get clustering res.
        H = np.asarray(H)
        indx_sort = np.argsort(H, axis=0)  # ascend order
        cluster_res = indx_sort[-1].reshape(-1)

        #  select band
        selected_band = []
        for c in np.unique(cluster_res):
            idx = np.nonzero(cluster_res == c)
            center = np.mean(X[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(X[:, idx[0]] - center, axis=0)
            band_ = X[:, idx[0]][:, distance.argmin()]
            selected_band.append(band_)
        while selected_band.__len__() < self.n_band:
            selected_band.append(np.zeros(X.shape[0]))
        bands = np.asarray(selected_band).transpose()
        return bands

    # # 得到W和H
    # def getWH(self, x_input, rank=10):
    #     snmf = nimfa.Snmf(x_input, seed="random_c", rank=rank, max_iter=12, version='r', eta=1.,
    #                         beta=1e-4, i_conv=10, w_min_change=0)
    #     snmf_fit = snmf()
    #     W = snmf.basis()
    #     H = snmf.coef()
    #     return W, H
    #
    # # 从H中选择每列最大的值
    # def maxh_selection(self, H):
    #     selection_h = []
    #     n_row, n_column = H.shape
    #     for i in range(n_column):
    #         max = H[0, i]
    #         for j in range(n_row-1):
    #             if H[j+1,i]>H[j,i]:
    #                 max = H[j+1, i]
    #         selection_h.append(max)
    #     return selection_h
    #
    # def fit(self, X):
    #     self.X = X
    #     return self
    #
    # # 波段选择，根据聚类选择中心波段
    # def predict(self, X):
    #     """
    #     Select band according to clustering center
    #     :param X: array like: shape (n_row, n_column, n_band)
    #     :return:
    #     """
    #     n_row, n_column, n_band = X.shape
    #     XX = X.reshape((n_row * n_column, -1))  # n_sample * n_band
    #     self.W, self.H = self.getWH(XX, rank=self.n_band)
    #     cluster_result = self.maxh_selection(self.H)
    #     selected_band = []
    #     n_cluster = np.unique(cluster_result).__len__()
    #     for c in np.unique(cluster_result):
    #         idx = np.nonzero(cluster_result == c)
    #         center = np.mean(XX[:, idx[0]], axis=1).reshape((-1, 1))
    #         distance = np.linalg.norm(XX[:, idx[0]] - center, axis=0)
    #         band_ = XX[:, idx[0]][:, distance.argmin()]
    #         selected_band.append(band_)
    #     bands = np.asarray(selected_band)
    #     bands = bands.reshape(n_cluster, n_row, n_column)
    #     bands = np.transpose(bands, axes=(1, 2, 0))
    #     return bands
    #
