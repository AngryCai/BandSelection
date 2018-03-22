from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from BandSelection.classes.DSC_NET import DSC_NET
import numpy as np
from BandSelection.classes.DSC_NET import DSCBS
from BandSelection.classes.SPEC import SPEC_HSI
from BandSelection.classes.utility import eval_band, eval_band_cv
from BandSelection.classes.SNMF import BandSelection_SNMF
from BandSelection.classes.Lap_score import Lap_score_HSI
from BandSelection.classes.NDFS import NDFS_HSI
from BandSelection.classes.ISSC import ISSC_HSI
from BandSelection.classes.SpaBS import SpaBS

if __name__ == '__main__':
    root = 'F:\Python\HSI_Files\\'
    #'/Users/cengmeng/PycharmProjects/python/Deep-subspace-clustering-networks/Data/'
    im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    # im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
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
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    X_img_2D = X_img.reshape(n_row * n_column, n_band)
    img_correct, gt_correct = p.get_correct(X_img, gt)
    gt_correct = p.standardize_label(gt_correct)
    train_inx, test_idx = p.get_tr_tx_index(gt_correct, test_size=0.4)

    n_input = [n_row, n_column]
    kernel_size = [7]
    n_hidden = [32]
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

    kwargs = {'n_input': n_input, 'n_hidden': n_hidden, 'reg_const1': reg1, 'reg_const2': reg2, 'max_iter':50,
              'kernel_size': kernel_size, 'batch_size': batch_size_test, 'model_path': model_path,
              'logs_path': logs_path}

    # algorithm = [DSCBS(n_selected_band, **kwargs)]
    alg_key = ['SPEC', 'Lap_score', 'NDFS', 'SpaBS', 'ISSC', 'SNMF', 'DSC']
    num_band = np.arange(5, 55, 5)
    mean_res, knn_res, svm_res, elm_res = [], [], [], []
    for n_selected_band in num_band:
        algorithm = [SPEC_HSI(n_selected_band),
                     Lap_score_HSI(n_selected_band),
                     NDFS_HSI(np.unique(gt_correct).shape[0], n_selected_band),
                     SpaBS(n_selected_band),
                     ISSC_HSI(n_selected_band, coef_=1.e-4),  #
                     BandSelection_SNMF(n_selected_band),
                     DSCBS(n_selected_band, **kwargs)]
        mean_score, knn_score, svm_score, elm_score = [], [], [], []
        for i in range(algorithm.__len__()):
            if i <= 4:
                X_new = algorithm[i].predict(X_img_2D)
                X_new, _ = p.get_correct(X_new.reshape(n_row, n_column, n_selected_band), gt)
            else:
                X_new = algorithm[i].predict(X_img)
                X_new, _ = p.get_correct(X_new, gt)
            knn, svm, elm, mean = eval_band_cv(X_new, gt_correct, times=10)
            mean_score.append(mean), knn_score.append(knn)
            svm_score.append(svm), elm_score.append(elm)
            print('n_band:%s, alg:%s\nknn: %s\nsvm: %s\nelm: %s\nmean:%s' % (n_selected_band, alg_key[i], knn, svm, elm, mean))
        mean_res.append(mean_score), knn_res.append(knn_score)
        svm_res.append(svm_score), elm_res.append(elm_score)
        print('-----------------------------------------')
        if n_selected_band % 3 == 0:
            np.savez('./score-indian.npz', mean_score=np.asarray(mean_res), knn_score=np.asarray(knn_res),
                     svm_score=np.asarray(svm_res), elm_score=np.asarray(elm_res))
    np.savez('./score-indian.npz', mean_score=np.asarray(mean_res), knn_score=np.asarray(knn_res),
             svm_score=np.asarray(svm_res), elm_score=np.asarray(elm_res))
