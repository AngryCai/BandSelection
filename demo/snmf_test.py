# coding:utf-8

from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from BandSelection.classes.SNMF import BandSelection_SNMF
import numpy as np

if __name__ == '__main__':
    root = '/Users/cengmeng/PycharmProjects/python/Deep-subspace-clustering-networks/Data/'
    # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    # im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    im_, gt_ = 'KSC', 'KSC_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    n_row, n_column, n_band = img.shape
    train_inx, test_idx = p.get_tr_tx_index(p.get_correct(img, gt)[1], test_size=0.9)

    img_train = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    img_train = np.transpose(img_train, axes=(2, 0, 1))  # Img.transpose()
    img_train = np.reshape(img_train, (n_band, n_row, n_column, 1))

    x_input = img.reshape(n_row * n_column, n_band)
    model_path = './pretrain-model-COIL20/model.ckpt'

    num_class = 7
    S = BandSelection_SNMF(x_input=x_input, model_path=model_path)

    w, h= S.getWH(x_input, num_class)
    y_x = S.maxh_selection(h)
    bands = S.band_selection(y_x, img)  # 选出每个类中的代表波段
    score = S.eval_band(bands, gt, train_inx, test_idx)  # 进行评价

    print('acc=%s' % score)

