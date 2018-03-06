from Toolbox.Preprocessing import Processor
from sklearn.preprocessing import minmax_scale
from BandSelection.classes.DSC_NET import DSC_BandSelection
import numpy as np

if __name__ == '__main__':
    root = 'F:\\Python\\HSI_Files\\'
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
    train_inx, test_idx = p.get_tr_tx_index(p.get_correct(img, gt)[1], test_size=0.9)
    img_train = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    img_train = np.transpose(img_train, axes=(2, 0, 1))  # Img.transpose()
    img_train = np.reshape(img_train, (n_band, n_row, n_column, 1))

    n_input = [n_row, n_column]
    kernel_size = [11]
    n_hidden = [16]*2
    batch_size = n_band
    model_path = './pretrain-model-Indian/model.ckpt'
    ft_path = './pretrain-model-Indian/model.ckpt'
    logs_path = './pretrain-model-Indian/logs'

    num_class = 5  # how many class we sample
    batch_size_test = n_band

    # test

    iter_ft = 0
    ft_times = 50
    display_step = 1
    alpha = 0.04
    learning_rate = 1e-3

    reg1 = 1e-4
    reg2 = 150.0

    dsc_bs = DSC_BandSelection(n_input=n_input, n_hidden=n_hidden, reg_const1=reg1, reg_const2=reg2, kernel_size=kernel_size,
                               batch_size=batch_size_test, model_path=model_path, logs_path=logs_path)

    acc_ = []
    all_loss = []
    all_acc = []
    for i in range(0, 1):
        # coil20_all_subjs = copy.deepcopy(Img)
        # coil20_all_subjs = coil20_all_subjs.astype(float)
        # label_all_subjs = copy.deepcopy(Label)
        # label_all_subjs = label_all_subjs - label_all_subjs.min() + 1
        # label_all_subjs = np.squeeze(label_all_subjs)
        dsc_bs.initlization()
        # CAE.restore()
        for iter_ft in range(ft_times):
            iter_ft = iter_ft + 1
            C, l1_cost, l2_cost, total_loss = dsc_bs.finetune_fit(img_train, learning_rate)
            all_loss.append(total_loss)
            if iter_ft % display_step == 0:
                print("epoch: %.1d" % iter_ft,
                      "L1 cost: %.8f, L2 cost: %.8f, total cost: %.8f" % (l1_cost, l2_cost, total_loss))
                C = dsc_bs.thrC(C, alpha)
                y_x, CKSym_x = dsc_bs.post_proC(C, num_class, 1, 4)
                bands = dsc_bs.band_selection(y_x, img)  # n_row * n_clm * n_class
                score = dsc_bs.eval_band(bands, gt, train_inx, test_idx)
                all_acc.append(score)
                print('eval score:', score)
        print(all_loss)
        print(all_acc)
