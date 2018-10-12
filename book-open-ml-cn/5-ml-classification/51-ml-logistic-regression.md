# 机器学习-51:逻辑回归分类算法(Logistic Regression)含代码

> [CSDN专栏: 机器学习理论与实践](https://blog.csdn.net/column/details/27839.html)

逻辑回归(Logistic Regression)分类算法属于监督学习算法。常用分类算法包括：逻辑回归(Logistic Regression, LR)、K最近邻(k-Nearest Neighbor, KNN)、朴素贝叶斯模型(Naive Bayesian Model, NBM)、隐马尔科夫模型(Hidden Markov Model)、支持向量机(Support Vector Machine)、决策树(Decision Tree)、神经网络(Neural Network)和集成学习(ada-boost)。

逻辑回归(Logistic Regression)解决问题的逻辑是：面对一个回归或者分类问题，建立代价函数，然后通过优化方法迭代求解出最优的模型参数，然后测试验证我们这个求解的模型的好坏。逻辑回归(Logistic Regression)可以用于回归或者分类问题。逻辑回归(Logistic Regression)分类算法能够解决二元分类和多元分类问题。

## 1 算法原理

线性回归的主要思想就是通过历史数据拟合出一条直线，用这条直线对新的数据进行预测。线性回归的公式如下：  
$z=θ_0+θ_1x_1+θ_2x_2+θ_3x_3...+θ_nx_n=θ^Tx$

逻辑回归(Logistic Regression)分类算法是将线性函数的结果映射到了sigmoid函数中。sigmoid函数的公式如下：$hθ(x)=1/(1+e^{-x})$

sigmoid的函数输出是介于(0，1)之间的，中间值是0.5; hθ(x)<0.5则说明当前数据属于A类; hθ(x)>0.5则说明当前数据属于B类。sigmoid函数看成样本数据的概率密度函数。逻辑回归(Logistic Regression)本质上也是线性回归。

逻辑回归(Logistic Regression)分类算法的核心步骤如下:

- 构造 predict 函数，一般采用Sigmoid函数;
- 构造 loss 函数, 一般采用对数损失函数
- 使用优化方法(梯度下降法、牛顿法等)最小化 loss 函数
- 反复迭代优化方法
- 输出分类类别

逻辑回归(Logistic Regression)分类算法的核心优势如下：

- 计算伸缩性: 基于线性回归，计算复杂度可控;
- 参数依赖性: 可调节参数较少;
- 普适性能力: 适用于连续型和离散型数据集；
- 抗噪音能力: 对缺失数据和异常数据比较敏感，需要特别关注;
- 结果解释性: 理论明确，解释性好。

## 2 算法实例

## 3 典型应用

在医学、社会学和统计学等方面有广泛用途。例如可以分析癌症和年龄之间的规律；可以分析早恋的社会学规律；可以分析个税收入的地域差异等。

## 系列文章

- [Gihutb专栏: 机器学习&深度学习(理论/实践)](https://github.com/media-tm/MTOpenML)
- [CSDN专栏: 机器学习理论与实践](https://blog.csdn.net/column/details/27839.html)
- [CSDN专栏: 深度学习理论与实践](https://blog.csdn.net/column/details/27839.html)

## 参考资料

- [1] 周志华. 机器学习. 清华大学出版社. 2016.
- [2] [日]杉山将. 图解机器学习. 人民邮电出版社. 2015.
- [3] 佩德罗·多明戈斯. 终极算法-机器学习和人工智能如何重塑世界. 中信出版社. 2018.
- [4] 李航. 统计学习方法. 2012.
- [5] [机器学习算法--逻辑回归原理介绍](https://blog.csdn.net/chibangyuxun/article/details/53148005)