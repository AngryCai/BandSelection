# coding:utf-8

from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from BandSelection.classes.SpaBS import SpaBS
import numpy as np
from BandSelection.classes.utility import eval_band


if __name__ == '__main__':
    root = 'F:\Python\HSI_Files\\'
    # root = '/Users/cengmeng/PycharmProjects/python/Deep-subspace-clustering-networks/Data/'
    im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    # im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    n_row, n_column, n_band = img.shape

    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    img_correct, gt_correct = p.get_correct(X_img, gt)
    train_inx, test_idx = p.get_tr_tx_index(gt_correct, test_size=0.4)
    # img_train = np.transpose(img_train, axes=(2, 0, 1))  # Img.transpose()
    # img_train = np.reshape(img_train, (n_band, n_row, n_column, 1))

    # x_input = img.reshape(n_row * n_column, n_band)
    model_path = './pretrain-model-COIL20/model.ckpt'

    n_select_band = 5

    spabs = SpaBS(n_select_band)
    X_new = spabs.predict(img_correct)  # 选出每个类中的代表波段

    # X_new, _ = p.get_correct(X_new, gt)        # 带入没有压缩的数据
    score = eval_band(X_new, gt_correct, train_inx, test_idx)  # 进行评价

    print('acc=%s' % score)

    # # ksvd测试
    # np.random.seed(0)
    # N = 1000
    # L = 64
    # n_features = 128
    # B = np.array(sp.sparse.random(N, L, density=0.5).todense())
    # D = np.random.randn(L, n_features)
    # X = np.dot(B, D)
    # dico = ApproximateKSVD(n_components=L, transform_n_nonzero_coefs=L)
    # dico.fit(X)
    # gamma = dico.transform(X)
    # assert_array_almost_equal(X, gamma.dot(dico.components_))
