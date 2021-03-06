# 重写Network 新增 train 和 update 函数
# 仅对 w5 和 w9 计算梯度并更新时的损失函数变化
# 作图
import numpy as np
from step2 import load_data
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生 w 的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.w[5] = -100.0
        self.w[9] = -100.0
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]  # 样本数
        cost = error * error
        cost = np.sum(cost) / num_samples
        print(cost)
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w5, gradient_w9, eta=0.01):
        self.w[5] = self.w[5] - eta * gradient_w5
        self.w[9] = self.w[9] - eta * gradient_w9

    def train(self, x, y, iterations=100, eta=0.01):
        points = []
        losses = []
        for i in range(iterations):
            points.append([self.w[5][0], self.w[9][0]])
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            gradient_w5 = gradient_w[5][0]
            gradient_w9 = gradient_w[9][0]
            self.update(gradient_w5, gradient_w9, eta)
            losses.append(L)
            # if i % 50 == 0:
            #     print('iter {}, point {}, loss {}'.format(i, [self.w[5][0], self.w[9][0]], L))
        return points, losses

if __name__ == '__main__':
    # 获取数据
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]
    # 创建网络
    net = Network(13)
    num_iterations = 2000
    # 启动训练
    points, losses = net.train(x, y, iterations=num_iterations)

    # 画出损失函数的变化趋势
    plot_x = np.arange(num_iterations)
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.title('Only w5 w9')
    plt.savefig('./step8.jpg')
    plt.show()