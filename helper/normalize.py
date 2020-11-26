import numpy as np


def normalize_x(data):
    '''
    标准化训练输入
    '''
    X_new = np.zeros(data[0].shape)
    feature_num = data[0].shape[1]
    X_norm = np.zeros((feature_num, 2))

    for i in range(feature_num):
        col_i = data[0][:, i]
        max_value = np.max(col_i)
        min_value = np.min(col_i)

        X_norm[i, 0] = min_value
        X_norm[i, 1] = max_value - min_value
        new_col = (col_i - X_norm[i, 0]) / X_norm[i, 1]
        X_new[:, i] = new_col

    return X_new, X_norm


def normalize_y(data):
    '''
    标准化训练标签
    '''
    Y_norm = np.zeros((1, 2))
    max_value = np.max(data[1])
    min_value = np.min(data[1])

    Y_norm[0, 0] = min_value
    Y_norm[0, 1] = max_value - min_value
    y_new = (data[1] - min_value) / Y_norm[0, 1]

    return y_new, Y_norm


def normalize_x_pred(data, X_norm):
    '''
    标准化预测输入
    '''
    X_new = np.zeros((data.shape))
    n = data.shape[1]

    for i in range(n):
        col_i = data[:, i]
        X_new[:, i] = (col_i - X_norm[i, 0]) / X_norm[i, 1]

    return X_new
