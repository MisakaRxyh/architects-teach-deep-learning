# 神经网络的训练
import numpy as np
from step2 import load_data
from step3 import Network

training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]

net = Network(13)
losses = []
# 只画出参数w5和w9在区间[-160, 160]的曲线部分，已经包含损失函数的极值
w5 = np.arange(-160.0, 160.0, 1.0)  # 从 -160.0 到 160.0 取值 每个数间隔1.0
w9 = np.arange(-160.0, 160.0, 1.0)  # 从 -160.0 到 160.0 取值 每个数间隔1.0
losses = np.zeros([len(w5), len(w9)])  # 建立一个零矩阵

# 计算设定区域内每个参数取值所对应的Loss
for i in range(len(w5)):
    for j in range(len(w9)):
        net.w[5] = w5[i]  # 从 -160.0 开始给神经网络的参数 w5 赋值 到 159.0
        net.w[9] = w9[j]
        z = net.forward(x)  # 改变参数 w5 和 w9 后的 线性回归 公式计算出 预测值z
        loss = net.loss(z, y)  # 计算 预测值z 和 真实值y 的损失loss
        losses[i, j] = loss  # 将参数 w5 和 w9 每一次变化产生的损失值保存到 损失矩阵losses 中

# 将两个变量和对应的Loss作3D图
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)
#
# w5, w9 = np.meshgrid(w5, w9)
#
# ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
# plt.show()

x1 = x[0]
y1 = y[0]
z1 = net.forward(x1)
print('x1 {}, shape {}'.format(x1, x1.shape))
print('y1 {}, shape {}'.format(y1, y1.shape))
print('z1 {}, shape {}'.format(z1, z1.shape))
gradient_w0 = (z1 - y1) * x1[0]
print('gradient_w0 {}'.format(gradient_w0))