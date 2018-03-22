"""
Description:
    auxiliary functions
"""
from Toolbox.Preprocessing import Processor
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import maxabs_scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
import numpy as np
import copy
from numpy import linalg
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


def eval_band(new_img, gt, train_inx, test_idx):
    p = Processor()
    # img_, gt_ = p.get_correct(new_img, gt)
    gt_ = gt
    img_ = maxabs_scale(new_img)
    # X_train, X_test, y_train, y_test = train_test_split(img_, gt_, test_size=0.4, random_state=42)
    X_train, X_test, y_train, y_test = img_[train_inx], img_[test_idx], gt_[train_inx], gt_[test_idx]
    knn_classifier = KNN(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    # score = cross_val_score(knn_classifier, img_, y=gt_, cv=3)
    y_pre = knn_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pre)
    # score = np.mean(score)
    return score


def eval_band_cv(X, y, times=10):
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    p = Processor()
    img_ = maxabs_scale(X)
    y_test_all = []
    estimator = [KNN(n_neighbors=5), SVC(C=1e4, kernel='rbf', gamma=1.), ELM_Classifier(200)]
    estimator_pre, y_test_all = [[], [], []], []
    for i in range(times):  # repeat N times K-fold CV
        skf = StratifiedKFold(n_splits=3, shuffle=True)
        for train_index, test_index in skf.split(img_, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_test_all.append(y_test)
            for c in range(3):
                estimator[c].fit(X_train, y_train)
                estimator_pre[c].append(estimator[c].predict(X_test))
    clf = ['knn', 'svm', 'elm']
    score = []
    for z in range(3):
        ca, oa, aa, kappa = p.save_res_4kfolds_cv(estimator_pre[z], y_test_all, file_name=clf[z] + 'score.npz', verbose=True)
        score.append([oa, kappa])
    return score


class ELM_Classifier(BaseEstimator, ClassifierMixin):
    upper_bound = .5
    lower_bound = -.5

    def __init__(self, n_hidden, dropout_prob=None):
        self.n_hidden = n_hidden
        self.dropout_prob = dropout_prob

    def fit(self, X, y, sample_weight=None):
        # check label has form of 2-dim array
        X, y, = copy.deepcopy(X), copy.deepcopy(y)
        self.sample_weight = None
        if y.shape.__len__() != 2:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.__len__()
            y = self.one2array(y, self.n_classes_)
        else:
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = self.classes_.__len__()
        self.W = np.random.uniform(self.lower_bound, self.upper_bound, size=(X.shape[1], self.n_hidden))
        if self.dropout_prob is not None:
            self.W = self.dropout(self.W, prob=self.dropout_prob)
            # X = self.dropout(X, prob=self.dropout_prob)
        self.b = np.random.uniform(self.lower_bound, self.upper_bound, size=self.n_hidden)
        H = expit(np.dot(X, self.W) + self.b)
        # H = self.dropout(H, prob=0.1)
        if sample_weight is not None:
            self.sample_weight = sample_weight / sample_weight.sum()
            extend_sample_weight = np.diag(self.sample_weight)
            inv_ = linalg.pinv(np.dot(
                np.dot(H.transpose(), extend_sample_weight), H))
            self.B = np.dot(np.dot(np.dot(inv_, H.transpose()), extend_sample_weight), y)
        else:
            self.B = np.dot(linalg.pinv(H), y)
        return self

    def one2array(self, y, n_dim):
        y_expected = np.zeros((y.shape[0], n_dim))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def predict(self, X, prob=False):
        X = copy.deepcopy(X)
        H = expit(np.dot(X, self.W) + self.b)
        output = np.dot(H, self.B)
        if prob:
            return output
        return output.argmax(axis=1)

    def get_params(self, deep=True):
        params = {'n_hidden': self.n_hidden, 'dropout_prob': self.dropout_prob}
        return params

    def set_params(self, **parameters):
        return self

    def dropout(self, x, prob=0.2):
        if prob < 0. or prob >= 1:
            raise Exception('Dropout level must be in interval [0, 1]')
        retain_prob = 1. - prob
        sample = np.random.binomial(n=1, p=retain_prob, size=x.shape)
        x *= sample
        # x /= retain_prob
        return x
