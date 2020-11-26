import numpy as np
from helper.net_type import *


class LossFunction(object):
    def __init__(self, net_type):
        self.net_type = net_type

    def check_loss(self, A, data):
        '''
        总选择函数
        '''
        if self.net_type == NetType.BinaryClassifier:
            loss = self.checkLoss_ce2(A, data)
        elif self.net_type == NetType.BinaryTanh:
            loss = self.checkLoss_ce2_tanh(A, data)
        return loss

    def checkLoss_ce2_tanh(self, A, data):
        '''
        修改后适用于 tanh 函数的损失函数
        '''
        p = (1 - data[1]) * np.log((1 - A) / 2) + \
            (1 + data[1]) * np.log((1 + A) / 2)
        LOSS = np.sum(-p)
        loss = LOSS / data[0].shape[0]
        return loss

    def checkLoss_ce2(self, A, data):
        '''
        交叉熵损失函数
        '''
        p = (data[1]) * np.log(A) + (1 - data[1]) * np.log(1 - A)
        LOSS = np.sum(-p)
        loss = LOSS / data[0].shape[0]
        return loss
