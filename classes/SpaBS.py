# coding:utf-8
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram




class ApproximateKSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements
        max_iter:
            Maximum number of iterations
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            I = gamma[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self, X):
        if min(X.shape) < self.n_components:
            D = np.random.randn(self.n_components, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, X):
        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._initialize(X)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = np.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)


class SpaBS(object):

    def __init__(self, n_band):
        self.n_band = n_band
        self.y_x = []

    def X_sort(self, X, C=0.5):
        """
        :param X:通过ksvd算法得到的系数矩阵
        :param Xs:按降序排列的前k个索引矩阵
        :return: 所有self的值
        """
        # 对系数矩阵每一行进行排序
        n_row, n_column = X.shape
        self.k = n_row * C
        self.Xs = []
        for i in range(n_column):
            # 矩阵self.X存储的是索引
            self.X[:, i] = sorted(np.argsort(-X[:, i]),reverse=True)
            self.Xs[:,i] = self.X[0:self.k, i]   #self.X[n_row-self.k:n_row-1, i]
        return self

    # 计算矩阵Xs的直方图，并对直方图降序进行排序，找出相应的k个谱带
    def Histogram_sort(self, d):
        """
        :param X: 计算出的矩阵Xs
        :param d:原始高光谱数据
        :return: k波段高光谱数据
        """
        h = []  # 直方图
        x = []  # 得出最后所求的高光谱数据
        idx = []
        row, column = self.Xs.shape
        # 计算直方图，并选出每一列前k个
        for i in range(column):
            myset = set(self.Xs[:, i])
            j = 0
            for item in myset:
                idx[j] = item
                h[j] = item * self.Xs[:, column].count(item)
                j = j + 1
            arr = np.argsort(h)[-self.k:][::-1]    # 找出前k个最大的数的下标
            x[:, column] = idx[arr]                # 将下标带入，得到原始波段号
        #     x[:, column] = list(set(x).intersection(set(d)))
        return x

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
        ### 使用SpaBS算法 ###
        # 调用ksvd
        dico = ApproximateKSVD(n_components=self.n_band, transform_n_nonzero_coefs=self.n_band)
        gamma = dico.transform(XX)  # gamma为系数矩阵
        # 得到k波段高光谱数据
        self.X_sort(gamma)
        self.y_x = self.Histogram_sort(X)

        cluster_result = self.y_x
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
        self.bands = np.transpose(bands, axes=(1, 2, 0))
        return self.bands





