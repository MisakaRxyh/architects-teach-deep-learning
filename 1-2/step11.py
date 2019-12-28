# 根据 step10 中的随机乱序和抽取部分小样本，改写训练过程
import numpy as np
from step2 import load_data
from step9 import Network
import matplotlib.pyplot as plt

# 获取数据
train_data, test_data = load_data()

# 打乱样本顺序
np.random.shuffle(train_data)

# 将train_data分成多个mini_batch
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]

# 创建网络
net = Network(13)

# 依次使用每个mini_batch的数据
for mini_batche in mini_batches:
    x = mini_batche[:, :-1]
    y = mini_batche[:, -1:]
    loss = net.train(x, y, iterations=1)
