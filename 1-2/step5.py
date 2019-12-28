# 计算梯度过程分析
import numpy as np
from step2 import load_data
from step3 import Network

# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1]  # training_data的前13列
y = training_data[:, -1:]  # training_data的最后1列

net = Network(13)
# 经过step4 参数 w5 和 w9 的值已经变成 159.0 了
# 相当于随机取了一点 159.0, 159.0 从这点开始找梯度最小值
net.w[5] = 159.0
net.w[9] = 159.0

# 取第一组数据
x1 = x[0]  # 前12维参数
y1 = y[0]  # 价格真实值
z1 = net.forward(x1)  # 价格预测值
print('x1 {}, shape {}'.format(x1, x1.shape))
print('y1 {}, shape {}'.format(y1, y1.shape))
print('z1 {}, shape {}'.format(z1, z1.shape))

# 计算 w0 的梯度
gradient_w0 = (z1 - y1) * x1[0]
print('gradient_w0 {}'.format(gradient_w0))
# 计算 w1 的梯度
gradient_w1 = (z1 - y1) * x1[1]
print('gradient_w1 {}'.format(gradient_w1))
# 计算 w2 的梯度
gradient_w2 = (z1 - y1) * x1[2]
print('gradient_w2 {}'.format(gradient_w2))

# gradient_w = []
#
# for i in range(13):
#     gradient_w.append((z1 - y1) * x1[i])
#
# print(gradient_w)

# 计算从 w0 到 w12 所有权重的梯度
# Numpy 广播机制
gradient_w = (z1 - y1) * x1  # ((1,) - (1,)) * (13,) = (13,)
# print(z1.shape, y1. shape, x1.shape)
print('gradient_w_by_sample1 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))

x2 = x[1]
y2 = y[1]
z2 = net.forward(x2)
gradient_w = (z2 - y2) * x2
print('gradient_w_by_sample2 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))

x3 = x[2]
y3 = y[2]
z3 = net.forward(x3)
gradient_w = (z3 - y3) * x3
print('gradient_w_by_sample3 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))

# 一次取三个样本数据，而不是取第三个样本
x3samples = x[0:3]
y3samples = y[0:3]
z3samples = net.forward(x3samples)
print('x {}, shape {}'.format(x3samples, x3samples.shape))
print('y {}, shape {}'.format(y3samples, y3samples.shape))
print('z {}, shape {}'.format(z3samples, z3samples.shape))
gradient_w = (z3samples - y3samples) * x3samples
print('gradient_w {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))

# 对于N个样本的情形，计算所有样本对梯度的贡献
# -> ste6