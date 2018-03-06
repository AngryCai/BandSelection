# coding:utf-8

from BandSelection.classes.DSC_HSI import ConvAE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # load face images and labels
    from Toolbox.Preprocessing import Processor
    from sklearn.preprocessing import minmax_scale

    root = '/Users/cengmeng/PycharmProjects/python/Deep-subspace-clustering-networks/Data/'
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
    # 归一化
    img_train = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    img_train = np.transpose(img_train, axes=(2, 0, 1))  # Img.transpose()
    img_train = np.reshape(img_train, (n_band, n_row, n_column, 1))

    n_input = [n_row, n_column]
    kernel_size = [17]
    n_hidden = [12]
    batch_size = n_band
    model_path = './pretrain-model-COIL20/model.ckpt'
    ft_path = './pretrain-model-COIL20/model.ckpt'
    logs_path = './pretrain-model-COIL20/logs'

    num_class = 7  # how many class we sample
    batch_size_test = n_band

    iter_ft = 0
    ft_times = 200
    display_step = 1
    alpha = 0.04
    learning_rate = 1e-3

    reg1 = 1e-4
    reg2 = 150.0

    CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_const1=reg1, reg_const2=reg2, kernel_size=kernel_size,
                 batch_size=batch_size_test, model_path=model_path, logs_path=logs_path)

    acc_ = []
    step = []
    finally_score = []
    for i in range(0, 1):

        CAE.initlization()

        for iter_ft in range(ft_times):
            iter_ft = iter_ft + 1
            C, l1_cost, l2_cost, total_loss = CAE.finetune_fit(img_train, learning_rate)   #根据梯度下降法算出最小化损失后的参数C
            if iter_ft % display_step == 0:
                step.append(iter_ft)
                print("epoch: %.1d" % iter_ft,
                      "L1 cost: %.8f, L2 cost: %.8f, total loss: %.8f" % (l1_cost, l2_cost, total_loss))
                C = CAE.thrC(C, alpha)    # 算出相似度矩阵C

                y_x, CKSym_x = CAE.post_proC(C, num_class, 1, 8)  # 得到进行谱聚类以后的结果
                bands = CAE.band_selection(y_x, img)      # 选出每个类中的代表波段
                score = CAE.eval_band(bands, gt, train_inx, test_idx)   # 进行评价
                finally_score.append(score)
                print('eval score:', score)

    plt.figure()
    plt.plot(step, finally_score)
    plt.xlabel('step')
    plt.ylabel('score')
    plt.title('score-step')
    plt.show()
