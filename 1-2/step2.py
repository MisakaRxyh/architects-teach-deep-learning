import numpy as np
import json


# 将step1中操作合并成load data函数
def load_data():
    # 读入训练数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前13项是影响因素，第14项是相应房屋的中位数价格
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)
    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 特征归一化
    # 计算训练集的最大值，最小值，平均值
    maximums = training_data.max(axis=0)
    minimums = training_data.min(axis=0)
    avgs = training_data.sum(axis=0) / training_data.shape[0]
    # 对数据进行归一化处理
    for i in range(feature_num):
        # print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


if __name__ == '__main__':
    # 获取数据
    training_data, test_data = load_data()
    # print(training_data)
    x = training_data[:, :-1]  # training_data的前13列
    y = training_data[:, -1:]  # training_data的最后1列
    # print(x)
    # print(y)
    # 参数权重
    w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
    w = np.array(w).reshape([13, 1])

    x1 = x[0]  # (1, 13)
    t = np.dot(x1, w)
    print(t)

    b = -0.2  # 偏移量
    z = t + b  # 完整的线性回归公式
    print(z)
