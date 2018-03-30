# coding:utf-8
from BandSelection.classes.utility import eval_band_cv
from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score
from BandSelection.classes.SNMF import BandSelection_SNMF
import numpy as np

if __name__ == '__main__':
    # root = '/Users/cengmeng/PycharmProjects/python/Deep-subspace-clustering-networks/Data/'
    root = 'F:\Python\HSI_Files\\'
    # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    n_row, n_column, n_band = img.shape
    train_inx, test_idx = p.get_tr_tx_index(p.get_correct(img, gt)[1], test_size=0.9)

    img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))

    x_input = img.reshape(n_row*n_column, n_band)

    num_class = 15
    snmf = BandSelection_SNMF(num_class)
    X_new = snmf.predict(x_input).reshape(n_row, n_column, num_class)
    a, b = p.get_correct(X_new, gt)
    b = p.standardize_label(b)
    print(eval_band_cv(a, b, times=5))

