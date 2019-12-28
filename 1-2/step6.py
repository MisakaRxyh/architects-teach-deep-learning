# 计算所有样本梯度值
import numpy as np
from step2 import load_data
from step3 import Network

if __name__ == '__main__':
    # 获取数据
    training_data, test_data = load_data()
    x = training_data[:, :-1]  # training_data的前13列
    y = training_data[:, -1:]  # training_data的最后1列

    net = Network(13)
    # 经过step4 参数 w5 和 w9 的值已经变成 159.0 了
    # 相当于随机取了一点 159.0, 159.0 从这点开始找梯度最小值
    net.w[5] = 159.0
    net.w[9] = 159.0

    # 对于N个样本的情形，计算所有样本对梯度的贡献
    z = net.forward(x)
    gradient_w = (z - y) * x
    # print('gradient_w shape {}'.format(gradient_w.shape))
    # print(gradient_w)

    # axis = 0 表示把每一行相加然后再除以总的行数
    gradient_w = np.mean(gradient_w, axis=0)
    print('gradient_w shape ', gradient_w.shape)
    print('w shape', net.w.shape)
    print('gradient_w', gradient_w)
    print('w', net.w)
    # 为了加减乘除等计算方便，gradient_w和w必须保持一致的形状
    gradient_w = gradient_w[:, np.newaxis]  # gradient.reshape(13,1)
    print('gradient_w shape', gradient_w.shape)

    # 计算梯度的代码整理如下
    z = net.forward(x)
    gradient_w = (z - y) * x
    gradient_w = np.mean(gradient_w, axis=0)
    gradient_w = gradient_w[:, np.newaxis]
    print(gradient_w)

    # 计算b的梯度的代码类似，如下
    gradient_b = (z - y)
    gradient_b = np.mean(gradient_b)
    # 此处b是一个数值，所以可以直接用np.mean得到一个标量
    print(gradient_b)


