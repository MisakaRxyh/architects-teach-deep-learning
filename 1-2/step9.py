# 重写 Network 重写 train 和 update 函数
# 对所有参数计算梯度并更新
import numpy as np
from step2 import load_data
import matplotlib.pyplot as plt


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生 w 的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w)
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]  # 样本数
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            # if (i + 1) % 10 == 0:
            #     print('iter {}, loss {}'.format(i, L))
        return losses

if __name__ == '__main__':
    # 获取数据
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]
    # 创建网络
    net = Network(13)
    num_iterations = 2000
    # 启动训练
    losses = net.train(x, y, num_iterations, eta=0.01)

    # 画出损失函数的变化趋势
    plot_x = np.arange(num_iterations)
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    # plt.title('All w')
    # plt.savefig('./step9.jpg')
    plt.show()

    # 注：与 step8 的区别在于 w 不同导致 z 不同
