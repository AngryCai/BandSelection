# coding:utf-8

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import nimfa
from sklearn.metrics import accuracy_score

class BandSelection_SNMF(object):
    def __init__(self, n_band):
        self.n_band = n_band

    # 得到W和H
    def getWH(self, x_input, rank=10):
        snmf = nimfa.Snmf(x_input, seed="random_c", rank=rank, max_iter=12, version='r', eta=1.,
                            beta=1e-4, i_conv=10, w_min_change=0)
        snmf_fit = snmf()
        W = snmf.basis()
        H = snmf.coef()
        return W, H

    # 从H中选择每列最大的值
    def maxh_selection(self, H):
        selection_h = []
        n_row, n_column = H.shape
        for i in range(n_column):
            max = H[0, i]
            for j in range(n_row-1):
                if H[j+1,i]>H[j,i]:
                    max = H[j+1, i]
            selection_h.append(max)
        return selection_h

    def fit(self, X):
        self.X = X
        return self

    # 波段选择，根据聚类选择中心波段
    def predict(self, X):
        """
        Select band according to clustering center
        :param X: array like: shape (n_row, n_column, n_band)
        :return:
        """
        n_row, n_column, n_band = X.shape
        XX = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        self.W, self.H = self.getWH(XX, rank=self.n_band)
        cluster_result = self.maxh_selection(self.H)
        selected_band = []
        n_cluster = np.unique(cluster_result).__len__()
        for c in np.unique(cluster_result):
            idx = np.nonzero(cluster_result == c)
            center = np.mean(XX[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(XX[:, idx[0]] - center, axis=0)
            band_ = XX[:, idx[0]][:, distance.argmin()]
            selected_band.append(band_)
        bands = np.asarray(selected_band)
        bands = bands.reshape(n_cluster, n_row, n_column)
        bands = np.transpose(bands, axes=(1, 2, 0))
        return bands

    # # 评价
    # def eval_band(self, new_img, gt, train_inx, test_idx):
    #     from Toolbox.Preprocessing import Processor
    #     from sklearn.neighbors import KNeighborsClassifier as KNN
    #     from sklearn.model_selection import cross_val_score
    #     from sklearn.preprocessing import maxabs_scale
    #     p = Processor()
    #
    #     # 处理数据，gt代表1-16的标签
    #     img_, gt_ = p.get_correct(new_img, gt)
    #     img_ = maxabs_scale(img_)
    #
    #     # 对数据进行处理，将数据分为训练集和测试集两类
    #     X_train, X_test, y_train, y_test = img_[train_inx], img_[test_idx], gt_[train_inx], gt_[test_idx]
    #     # X_train, X_test, y_train, y_test = train_test_split(img_, gt_, test_size=0.9, random_state=42)
    #
    #     ## 使用KNN进行聚类 ##
    #     knn_classifier = KNN(n_neighbors=5)
    #     knn_classifier.fit(X_train, y_train)
    #     y_pre = knn_classifier.predict(X_test)
    #
    #     # ## 使用SVM进行聚类 ##
    #     # from sklearn.svm import SVC, LinearSVC
    #     #
    #     # # svm_classifier = SVC()
    #     # svm_classifier = SVC(kernel='rbf', gamma=120.0, C=1)  #径向核函数
    #     # # svm_classifier = SVC(kernel='poly', degree=3, C=1)  #多项式核函数
    #     # # svm_classifier = LinearSVC(C=1)     # 线性核函数
    #     #
    #     # svm_classifier.fit(X_train, y_train)
    #     # y_pre = svm_classifier.predict(X_test)
    #
    #     # 得到测试精度
    #     finally_score = accuracy_score(y_test, y_pre)
    #
    #     # 交叉验证
    #     # score = cross_val_score(knn_classifier, img_, y=gt_, cv=3)
    #     # score_mean = np.mean(score)
    #     return finally_score

