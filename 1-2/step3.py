import numpy as np
from step2 import load_data
import json


# 构建神经网络
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    # new 1
    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost


if __name__ == '__main__':
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]

    net = Network(13)
    x1 = x[0:3]
    y1 = y[0:3]
    z = net.forward(x1)
    print('predict:', z)
    loss = net.loss(z, y1)
    print('loss:', loss)
