# 小批量随机梯度下降法（Mini-batch Stochastic Gradient Descent）
# 在之前的step中，每次迭代的时候均基于数据集中的全部数据进行计算。
# 但在实际问题中数据集往往非常大，如果每次计算都使用全部的数据来计算损失函数和梯度，效率非常低。
# 一个合理的解决方案是每次从总的数据集中随机抽取出小部分数据来代表整体，基于这部分数据计算梯度和损失，然后更新参数。
# 这种方法被称作小批量随机梯度下降法（Mini-batch Stochastic Gradient Descent），简称SGD。
# 每次迭代时抽取出来的一批数据被称为一个min-batch，一个mini-batch所包含的样本数目称为batch_size。
# 当程序迭代的时候，按mini-batch逐渐抽取出样本，当把整个数据集都遍历到了的时候，则完成了一轮的训练，也叫一个epoch。
# 启动训练时，可以将训练的轮数num_epochs和batch_size作为参数传入。

import numpy as np
from step2 import load_data
from step9 import Network
# 获取数据
train_data, test_data = load_data()
print(train_data.shape)

train_data1 = train_data[:10]
print(train_data1.shape)

# 使用train_data1的数据（0-9号样本）计算梯度并更新网络参数
net = Network(13)
x = train_data1[:, :-1]
y = train_data1[:, -1:]
loss = net.train(x, y, iterations=1,eta=0.01)
print(loss)

# 再取出10-19号样本作为第二个mini-batch，计算梯度并更新网络参数。
train_data2 = train_data[10:19]
x = train_data2[:, :-1]
y = train_data2[:, -1:]
loss = net.train(x, y, iterations=1)
print(loss)

# 按此方法不断的取出新的mini-batch并逐渐更新网络参数。
# 下面的程序可以将train_data分成大小为batch_size的多个mini_batch。
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
print('total number of mini_batches is ', len(mini_batches))
print('first mini_batch shape ', mini_batches[0].shape)
print('last mini_batch shape ', mini_batches[-1].shape)

# 随机打乱样本顺序 np.random.shuffle函数
# 新建一个array
a = np.array([k+1 for k in range(12)])
print('before shuffle', a)
np.random.shuffle(a)
print('after shuffle', a)

# 新建一个array
a = np.array([k+1 for k in range(12)])
a = a.reshape([6, 2])
print('before shuffle', a)
np.random.shuffle(a)
print('after shuffle', a)

