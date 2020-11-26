from helper.data_handler import *
import math


class Network(object):
    def __init__(self, sizes, params):
        input_size = sizes[0]
        output_size = sizes[1]
        self.W = np.zeros((input_size, output_size))
        self.B = np.zeros((1, output_size))

        self.eta = params["eta"]
        self.batch_size = params["batch_size"]
        self.max_epoch = params["max_epoch"]
        self.eps = params["eps"]

    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        return Z

    def __checkLoss_mse(self, data):
        '''
        均方差损失函数
        '''
        X, Y = data[0], data[1]
        m = X.shape[0]

        Z = self.__forwardBatch(X)
        LOSS = (Z - Y) ** 2
        loss = LOSS.sum() / (m * 2)
        return loss

    def __backwardBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y

        dW = np.dot(batch_x.T, dZ) / m
        dB = dZ.sum(axis=0, keepdims=True) / m
        return dW, dB

    def __update(self, dW, dB):
        self.W -= self.eta * dW
        self.B -= self.eta * dB

    def train(self, data, check_point=0.1):
        loss = 10
        max_iteration = math.ceil(len(data[0]) / self.batch_size)
        checkpoint_iteration = (int)(max_iteration * check_point)

        for epoch in range(self.max_epoch):
            shuffle_data = shuffle(data)
            for iteration in range(max_iteration):
                batch_x, batch_y = get_batch(
                    self.batch_size, iteration, shuffle_data)

                batch_z = self.__forwardBatch(batch_x)
                dW, dB = self.__backwardBatch(batch_x, batch_y, batch_z)
                self.__update(dW, dB)

                total_iteration = epoch * max_iteration + iteration + 1
                if(total_iteration % checkpoint_iteration == 0):
                    loss = self.__checkLoss_mse(data)
                    print(
                        f"Epoch: {epoch}, iteration: {iteration}, loss: {loss}, W: {self.W}, B: {self.B}")
                    if(loss < self.eps):
                        break
            if(loss < self.eps):
                break

        print(f"Finished\n------------------\nW: {self.W}\nB: {self.B}")

    def inference(self, x):
        return self.__forwardBatch(x)
