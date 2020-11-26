import numpy as np


class Sigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a


class Tanh(object):
    def forward(self, z):
        a = 2.0 / (1.0 + np.exp(-2 * z)) - 1.0
        return a
