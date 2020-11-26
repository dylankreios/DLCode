from helper.classify import *


class DataReader(object):
    def __init__(self):
        pass

    def read_logic_not(self):
        X = np.array([0, 1]).reshape(2, 1)
        Y = np.array([1, 0]).reshape(2, 1)
        return X, Y

    def read_logic_and(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([0, 0, 0, 1]).reshape(4, 1)
        return X, Y

    def read_logic_nand(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([1, 1, 1, 0]).reshape(4, 1)
        return X, Y

    def read_logic_or(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([0, 1, 1, 1]).reshape(4, 1)
        return X, Y

    def read_logic_nor(self):
        X = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
        Y = np.array([1, 0, 0, 0]).reshape(4, 1)
        return X, Y


if __name__ == "__main__":
    net = Network([2, 1], {"eta": 0.5, "batch_size": 1,
                           "max_epoch": 4236, "eps": 2e-3})
    reader = DataReader()
    data = reader.read_logic_and()
    net.train(data, check_point=1)
