# Code Authors: Pan Ji,     University of Adelaide,         pan.ji@adelaide.edu.au
#               Tong Zhang, Australian National University, tong.zhang@anu.edu.au
# Copyright Reserved!
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
from sklearn import cluster
from munkres import Munkres
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from tensorflow.examples.tutorials.mnist import input_data
import copy

class DSC_BandSelection(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_const1=1.0, reg_const2=1.0, reg=None, batch_size=256, \
                 denoise=False, model_path=None, logs_path='./logs'):
        # n_hidden is a arrary contains the number of neurals on every layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg
        self.model_path = model_path
        self.kernel_size = kernel_size
        self.iter = 0
        self.batch_size = batch_size
        weights = self._initialize_weights()

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])
        self.learning_rate = tf.placeholder(tf.float32, [])

        if denoise == False:
            x_input = self.x
            latent, shape = self.encoder(x_input, weights)

        else:
            x_input = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                      mean=0,
                                                      stddev=0.2,
                                                      dtype=tf.float32))

            latent, shape = self.encoder(x_input, weights)
        self.z_conv = tf.reshape(latent, [batch_size, -1])
        self.z_ssc, Coef = self.selfexpressive_moduel(batch_size)
        self.Coef = Coef
        latent_de_ft = tf.reshape(self.z_ssc, tf.shape(latent))
        self.x_r_ft = self.decoder(latent_de_ft, weights, shape)

        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])

        self.cost_ssc = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.z_conv, self.z_ssc), 2))
        self.recon_ssc = tf.reduce_sum(tf.pow(tf.subtract(self.x_r_ft, self.x), 2.0))
        self.reg_ssc = tf.reduce_sum(tf.pow(self.Coef, 2))
        tf.summary.scalar("ssc_loss", self.cost_ssc)
        tf.summary.scalar("reg_lose", self.reg_ssc)

        self.loss_ssc = self.cost_ssc * reg_const2 + reg_const1 * self.reg_ssc + self.recon_ssc

        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer_ssc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_ssc)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['enc_w0'] = tf.get_variable("enc_w0",
                                                shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

        all_weights['dec_w0'] = tf.get_variable("dec_w0",
                                                shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
        all_weights['dec_b0'] = tf.Variable(tf.zeros([1], dtype=tf.float32))
        return all_weights

    # Building the encoder
    def encoder(self, x, weights):
        shapes = []
        # Encoder Hidden layer with relu activation #1
        shapes.append(x.get_shape().as_list())
        layer1 = tf.nn.bias_add(tf.nn.conv2d(x, weights['enc_w0'], strides=[1, 2, 2, 1], padding='SAME'),
                                weights['enc_b0'])
        layer1 = tf.nn.relu(layer1)
        return layer1, shapes

    # Building the decoder
    def decoder(self, z, weights, shapes):
        # Encoder Hidden layer with relu activation #1
        shape_de1 = shapes[0]
        layer1 = tf.add(tf.nn.conv2d_transpose(z, weights['dec_w0'], tf.stack(
            [tf.shape(self.x)[0], shape_de1[1], shape_de1[2], shape_de1[3]]), \
                                               strides=[1, 2, 2, 1], padding='SAME'), weights['dec_b0'])
        layer1 = tf.nn.relu(layer1)

        return layer1

    def selfexpressive_moduel(self, batch_size):

        Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')
        z_ssc = tf.matmul(Coef, self.z_conv)
        return z_ssc, Coef

    def finetune_fit(self, X, lr):
        C, l1_cost, l2_cost, total_loss, summary, _ = self.sess.run(
            (self.Coef, self.reg_ssc, self.cost_ssc, self.loss_ssc, self.merged_summary_op, self.optimizer_ssc), \
            feed_dict={self.x: X, self.learning_rate: lr})
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return C, l1_cost, l2_cost, total_loss

    def initlization(self):
        tf.reset_default_graph()
        self.sess.run(self.init)

    def transform(self, X):
        return self.sess.run(self.z_conv, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.model_path)
        print("model restored")

    def best_map(self, L1, L2):
        # L1 should be the labels and L2 should be the clustering number we got
        Label1 = np.unique(L1)
        nClass1 = len(Label1)
        Label2 = np.unique(L2)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = L1 == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = L2 == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        newL2 = np.zeros(L2.shape)
        for i in range(nClass2):
            newL2[L2 == Label2[i]] = Label1[c[i]]
        return newL2

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        n = C.shape[0]
        C = 0.5 * (C + C.T)
        C = C - np.diag(np.diag(C)) + np.eye(n,
                                             n)  # for sparse C, this step will make the algorithm more numerically stable
        r = d * K + 1
        print('r = %s, C:%s' % (r, np.unique(C).shape))
        U, S, _ = svds(C, r, v0=np.ones(n))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L)
        return grp, L

    def band_selection(self, cluster_result, img):
        """
        Select band according to clustering center
        :param cluster_result:
        :param img: array like: shape (n_row, n_column, n_band)
        :return:
        """
        selected_band = []
        n_row, n_column, n_band = img.shape
        n_cluster = np.unique(cluster_result).__len__()
        img_ = img.reshape((n_row * n_column, -1))  # n_sample * n_band
        for c in np.unique(cluster_result):
            idx = np.nonzero(cluster_result == c)
            center = np.mean(img_[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(img_[:, idx[0]] - center, axis=0)
            band_ = img_[:, idx[0]][:, distance.argmin()]
            selected_band.append(band_)
        bands = np.asarray(selected_band)
        bands = bands.reshape(n_cluster, n_row, n_column)
        bands = np.transpose(bands, axes=(1, 2, 0))
        return bands

    def eval_band(self, new_img, gt, train_inx, test_idx):
        from Toolbox.Preprocessing import Processor
        from sklearn.neighbors import KNeighborsClassifier as KNN
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import maxabs_scale
        from sklearn.metrics import accuracy_score
        p = Processor()
        img_, gt_ = p.get_correct(new_img, gt)
        img_ = maxabs_scale(img_)
        # X_train, X_test, y_train, y_test = train_test_split(img_, gt_, test_size=0.4, random_state=42)
        X_train, X_test, y_train, y_test = img_[train_inx], img_[test_idx], gt_[train_inx], gt_[test_idx]
        knn_classifier = KNN(n_neighbors=5)
        knn_classifier.fit(X_train, y_train)
        # score = cross_val_score(knn_classifier, img_, y=gt_, cv=3)
        y_pre = knn_classifier.predict(X_test)
        score = accuracy_score(y_test, y_pre)
        # score = np.mean(score)
        return score

    def err_rate(self, gt_s, s):
        c_x = self.best_map(gt_s, s)
        err_x = np.sum(gt_s[:] != c_x[:])
        missrate = err_x.astype(float) / (gt_s.shape[0])
        return missrate

