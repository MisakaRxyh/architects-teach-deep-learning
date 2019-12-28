import numpy as np
import json

# 读入训练数据
datafile = './work/housing.data'
data = np.fromfile(datafile, sep=' ')
# print(data.shape)

# 读入之后的数据被转化成1维array，其中array的
# 第0-13项是第一条数据，第14-27项是第二条数据，....
# 这里对原始数据做reshape，变成N x 14的形式
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])
x = data[0]
print(data.shape)
print(x)

# 取前80%作为训练集
ratio = 0.8
offset = int(data.shape[0] * ratio)
training_data = data[:offset]
print(training_data.shape)

# 特征归一化
# 计算训练集的最大值，最小值，平均值
maximums = training_data.max(axis=0)
minimums = training_data.min(axis=0)
avgs = training_data.sum(axis=0) / training_data.shape[0]
# 对数据进行归一化处理
for i in range(feature_num):
    print(maximums[i], minimums[i], avgs[i])
    data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

