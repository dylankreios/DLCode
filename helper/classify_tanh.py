from helper.classify import *
from helper.classify_function import *
from helper.net_type import *


class TanhNetwork(Network):
    def forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        if self.net_type == NetType.BinaryClassifier:
            A = Sigmoid().forward(Z)
            return A
        elif self.net_type == NetType.BinaryTanh:
            A = Tanh().forward(Z)
            return A
        else:
            return Z

    def backwardBatch(self, batch_x, batch_y, batch_a):
        m = batch_x.shape[0]
        dZ = 2 * (batch_a - batch_y)

        dB = dZ.sum(axis=0, keepdims=True) / m
        dW = np.dot(batch_x.T, dZ) / m
        return dW, dB
