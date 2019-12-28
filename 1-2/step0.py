# 重写Network 更新 train 函数
# 实现 SGD 算法的最终 train 函数
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

    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        # num_epochs 训练轮数
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
            # 然后再按每次取 batch_size 条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个 mini_batch 包含 batch_size 条数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                z = self.forward(x)
                loss = self.loss(z, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))
        return losses


if __name__ == '__main__':
    # 获取数据
    train_data, test_data = load_data()

    # 创建网络
    net = Network(13)
    # 启动训练
    losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)

    # 画出损失函数的变化趋势
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.title('SDG')
    plt.savefig('./step0.jpg')
    plt.show()
