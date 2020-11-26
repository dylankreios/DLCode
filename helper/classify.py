import math
from helper.data_handler import *
from helper.classify_function import *
from helper.net_type import *
from helper.loss_function import *


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

        self.net_type = params["net_type"]

    def forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        if self.net_type == NetType.BinaryClassifier:
            A = Sigmoid().forward(Z)
            return A
        else:
            return Z

    def backwardBatch(self, batch_x, batch_y, batch_a):
        m = batch_x.shape[0]
        dZ = batch_a - batch_y

        dW = np.dot(batch_x.T, dZ) / m
        dB = dZ.sum(axis=0, keepdims=True) / m
        return dW, dB

    def __update(self, dW, dB):
        self.W -= self.eta * dW
        self.B -= self.eta * dB

    def train(self, data, check_point=0.1):
        loss = 10
        loss_function = LossFunction(self.net_type)
        max_iteration = math.ceil(len(data[0]) / self.batch_size)
        checkpoint_iteration = (int)(max_iteration * check_point)

        for epoch in range(self.max_epoch):
            shuffle_data = shuffle(data)
            for iteration in range(max_iteration):
                batch_x, batch_y = get_batch(
                    self.batch_size, iteration, shuffle_data)

                batch_a = self.forwardBatch(batch_x)
                dW, dB = self.backwardBatch(batch_x, batch_y, batch_a)
                self.__update(dW, dB)

                total_iteration = epoch * max_iteration + iteration + 1
                if(total_iteration % checkpoint_iteration == 0):
                    loss = self.__check_loss(loss_function, data)
                    print(
                        f"Epoch: {epoch}, iteration: {iteration}, loss: {loss}, W: {self.W}, B: {self.B}")
                    if(loss < self.eps):
                        break
            if(loss < self.eps):
                break

        print(f"Finished\n------------------\nW: {self.W}\nB: {self.B}")

    def inference(self, x):
        return self.forwardBatch(x)

    def __check_loss(self, loss_func, data):
        A = self.forwardBatch(data[0])
        loss = loss_func.check_loss(A, data)
        return loss
