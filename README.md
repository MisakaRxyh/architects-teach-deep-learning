# architects-teach-deep-learning
深度学习学习笔记
## 1-1节 深度学习与机器学习综述

## 1-2节 用Python编写 房价预测 模型
### 总结
本节，我们详细讲解了如何使用numpy实现梯度下降算法，构建并训练了一个简单的线性模型实现波士顿房价预测，可以总结出，使用神经网络建模房价预测有三个要点：

+ 构建网络，初始化参数w和b，定义预测和损失函数的计算方法。
+ 随机选择初始点，建立梯度的计算方法，和参数更新方式。
+ 从总的数据集中抽取部分数据作为一个mini_batch，计算梯度并更新参数，不断迭代直到损失函数几乎不再下降。

### 自己的话
作为深度学习的HelloWorld程序，此案例的要点在于把握线性回归的公式定义，根据公式拟合预测出房价，**重点在于不断优化参数$w$与$b$使得真实值与预测值的差距尽可能的小**。    
这里的训练，就是根据损失函数以及梯度的定义，不断的循环，找出最小的那个损失函数的值，而此时的参数$w$与$b$就是模型效果好的参数
梯度的定义：
> 梯度的是一个向量，梯度的方向是方向导数中取到最大值的方向，**方向导数**是各个方向上的导数     
> 可以理解为梯度的方向就是函数上升的方向

### 本节的公式：
#### 线性回归模型
$$y = {\sum_{j=1}^Mx_j w_j} + b$$
$w_j$和$b$分别表示该线性模型的权重和偏置。一维情况下，$w_j$和$b$就是直线的斜率和截距。

#### 损失函数
$$Loss = (y - z)^2$$
$$L= \frac{1}{N}\sum_i{(y^{(i)} - z^{(i)})^2}$$

#### 计算梯度

上面我们讲过了损失函数的计算方法，这里稍微加以改写，引入因子$\frac{1}{2}$，定义损失函数如下
$$L= \frac{1}{2N}\sum_{i=1}^N{(y^{(i)} - z^{(i)})^2}$$
其中$z_i$是网络对第$i$个样本的预测值
$$z^{(i)} = \sum_{j=0}^{12}{x_j^{(i)} w^{(j)}} + b$$

可以计算出$L$对$w$和$b$的偏导数

$$\frac{\partial{L}}{\partial{w_j}} = \frac{1}{N}\sum_i^N{(z^{(i)} - y^{(i)})\frac{\partial{z^{(i)}}}{w_j}} = \frac{1}{N}\sum_i^N{(z^{(i)} - y^{(i)})x_j^{(i)}}$$

$$\frac{\partial{L}}{\partial{b}} = \frac{1}{N}\sum_i^N{(z^{(i)} - y^{(i)})\frac{\partial{z^{(i)}}}{b}} = \frac{1}{N}\sum_i^N{(z^{(i)} - y^{(i)})}$$

#### 梯度计算公式
$$\frac{\partial{L}}{\partial{w_j}} = \frac{1}{N}\sum_i^N{(z^{(i)} - y^{(i)})\frac{\partial{z_j^{(i)}}}{w_j}} = \frac{1}{N}\sum_i^N{(z^{(i)} - y^{(i)})x_j^{(i)}}$$

