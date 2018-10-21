# Tensorflow的回归算法

> [机器学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83030331)

如果想切换数据集，用其他数据集折腾以下，请从[kaggle/datasets](https://www.kaggle.com/datasets)下载。本文使用tensorflow.python.keras自带的波斯顿房价回归数据集作为示例。boston_housing数据集的数据来自1970年代波斯顿周边地区的房价，是用于机器学习的经典数据集。该数据集共计506条数据记录，分为404个训练样本和102个测试样本，每条数据包含13个特征。

## 0 Keras的优化器和损失函数

Keras提供了一些优化器供选择，包括: Stochastic Gradient Descent(SGD), Adam, RMSprop, AdaGrad和AdaDelta等。对于大多数问题，RMSprop优化器是一个很好的选择。

Keras提供了一些损失函数供选择。在监督学习问题中，我们必须找到实际值和预测值之间的误差。 可以使用不同的度量标准来评估此错误。 该度量通常称为损失函数或成本函数或目标函数。 根据您对错误的处理方式，可以有多个损失函数。 Keras的损失函数包括: binary-cross-entropy 用于二元分类问题；categorical-cross-entropy 用于多类分类问题；mean-squared-error用于回归问题。

## 1 Tensorflow的线性回归(Linear Regression)

### 1.1 构建线性回归模型

```python
from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
```

### 1.2 训练线性回归模型

### 1.3 调试线性回归模型

## 2 Tensorflow的多项式回归(Polynomial Regression)

### 2.1 构建多项式回归模型

```python
from tensorflow.python import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
```

### 2.2 训练多项式回归模型

### 2.3 调试多项式回归模型

## 系列文章

- [深度学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83040730)
- [机器学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83030331)
- [Github: 机器学习&深度学习理论与实践(开源图书)](https://github.com/media-tm/MTOpenML)

## 参考资料

- [1] 周志华. 机器学习. 清华大学出版社. 2016.
- [2] [日]杉山将. 图解机器学习. 人民邮电出版社. 2015.
- [3] 佩德罗·多明戈斯. 终极算法-机器学习和人工智能如何重塑世界. 中信出版社. 2018.
- [4] [Tensorflow官网](https://www.tensorflow.org/)