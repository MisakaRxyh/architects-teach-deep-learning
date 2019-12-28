# 重写Network 新增 gradient 函数 求每个参数梯度
# 寻找损失函数更小的点
import numpy as np
from step2 import load_data


# 将计算 w 和 b 梯度的过程，写成Network类的gradient函数
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生 w 的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0

    def forword(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forword(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)

        return gradient_w, gradient_b

if __name__ == '__main__':
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]

    # 调用上面定义的gradient函数，计算梯度
    # 初始化网络
    net = Network(13)
    # 设置[w5, w9] = [-100., +100.]
    net.w[5] = -100.0
    net.w[9] = -100.0

    z = net.forword(x)
    loss = net.loss(z, y)
    gradient_w, gradient_b = net.gradient(x, y)
    gradient_w5 = gradient_w[5][0]
    gradient_w9 = gradient_w[9][0]
    print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
    print('gradient {}'.format([gradient_w5, gradient_w9]))

    # 在 [w5, w9] 平面上，沿着梯度的反方向移动到下一个点P1
    # 定义移动步长 eta
    eta = 0.1
    # 更新参数 w5 和 w9
    net.w[5] = net.w[5] - eta * gradient_w5
    net.w[9] = net.w[9] - eta * gradient_w9
    # 重新计算 z 和 loss
    z = net.forword(x)
    loss = net.loss(z, y)
    gradient_w, gradient_b = net.gradient(x, y)
    gradient_w5 = gradient_w[5][0]
    gradient_w9 = gradient_w[9][0]
    print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
    print('gradient {}'.format([gradient_w5, gradient_w9]))