import numpy as np


def get_batch(batch_size, iteration, data):
    '''
    获取批量数据
    '''
    start = iteration * batch_size
    end = start + batch_size

    batch_x = data[0][start:end, :]
    batch_y = data[1][start:end, :]
    return batch_x, batch_y


def shuffle(data):
    '''
    随机打乱数据
    '''
    seed = np.random.randint(0, 100)

    np.random.seed(seed)
    XP = np.random.permutation(data[0])
    np.random.seed(seed)
    YP = np.random.permutation(data[1])

    return XP, YP
