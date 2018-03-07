from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from BandSelection.classes.DSC_NET import DSC_NET
import numpy as np
from BandSelection.classes.DSC_NET import DSCBS
from BandSelection.classes.SPEC import SPEC_HSI
from BandSelection.classes.utility import eval_band
from BandSelection.classes.SNMF import BandSelection_SNMF


if __name__ == '__main__':
    root = 'F:\Python\HSI_Files\\'
    #'/Users/cengmeng/PycharmProjects/python/Deep-subspace-clustering-networks/Data/'

    # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    # im_, gt_ = 'Botswana', 'Botswana_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    # Img, Label = Img[:256, :, :], Label[:256, :]
    n_row, n_column, n_band = img.shape
    train_inx, test_idx = p.get_tr_tx_index(p.get_correct(img, gt)[1], test_size=0.4)
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))


    n_input = [n_row, n_column]
    kernel_size = [11]
    n_hidden = [16]
    batch_size = n_band
    model_path = './pretrain-model-COIL20/model.ckpt'
    ft_path = './pretrain-model-COIL20/model.ckpt'
    logs_path = './pretrain-model-COIL20/logs'

    batch_size_test = n_band

    iter_ft = 0
    display_step = 1
    alpha = 0.04
    learning_rate = 1e-3

    reg1 = 1e-4
    reg2 = 150.0

    n_selected_band = 5

    kwargs = {'n_input': n_input, 'n_hidden': n_hidden, 'reg_const1': reg1, 'reg_const2': reg2, 'max_iter':10,
              'kernel_size': kernel_size, 'batch_size': batch_size_test, 'model_path': model_path,
              'logs_path': logs_path}

    algorithm = [#SPEC_HSI(n_selected_band),
                 BandSelection_SNMF(n_selected_band),
                 DSCBS(n_selected_band, **kwargs)
                 ]

    for alg in algorithm:
        alg.fit(X_img)
        X_new = alg.predict(X_img)
        acc = eval_band(X_new, gt, train_inx, test_idx)
        print(acc)
