# 机器学习-50:分类算法概述

> [机器学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83030331)

机器学习分为监督学习、无监督学习和半监督学习(强化学习)。无监督学习最常应用的场景是聚类(clustering)和降维(dimension reduction)。分类算法和回归算法都属于监督学习算法, 其中分类算法的目标就是：学习数据集的数据特征，并将原始数据特征映射到目标的分类类别。分类算法包括：逻辑回归(Logistic Regression, LR)、K最近邻(k-Nearest Neighbor, KNN)、朴素贝叶斯模型(Naive Bayesian Model, NBM)、隐马尔科夫模型(Hidden Markov Model)、支持向量机(Support Vector Machine)、决策树(Decision Tree)、神经网络(Neural Network)和集成学习(ada-boost)。其中集成学习(ada-boost)是一个混合分类方法。

> 告别碎片阅读，构成知识谱系。一起阅读和完善: [机器学习原理与实践(开源图书)](https://github.com/media-tm/MTOpenML)

## 如何构建分类算法

构建通用分类算法模型的步骤如下:

- 初步评估数据集的数据特征或了解其先验特征; 
- 选择分类算法模型, 分类算法工具箱里包含：逻辑回归(Logistic Regression, LR)、K最近邻(k-Nearest Neighbor, KNN)、朴素贝叶斯模型(Naive Bayesian Model, NBM)、隐马尔科夫模型(Hidden Markov Model)、支持向量机(Support Vector Machine)、决策树(Decision Tree)、神经网络(Neural Network)和集成学习(ada-boost)。
- 构建预测(Predict)函数, 预测函数的选择和使用的分类模型息息相关，此处不再展开。
- 构建损失(Loss)函数，损失函数工具箱包括: 0-1损失函数(0-1 loss function)、平方损失函数(quadratic loss function)、绝对值损失函数(absolute loss function)、对数损失函数(logarithmic loss function) 和 对数似然损失函数(log-likehood loss function)。
- 选择优化函数，使用优化函数最小化损失(Loss)函数。常有优化函数包括：梯度下降法、牛顿法等。
- 反复迭代优化方法
- 输出分类类别

## 评估分类算法的方法

用来比较和评估分类方法的标准主要有：

- 预测的准确率。模型正确地预测新样本的类标号的能力;
- 计算速度。包括构造模型以及使用模型进行分类的时间;
- 强壮性。模型对噪声数据或空缺值数据正确预测的能力;
- 可伸缩性。对于数据量很大的数据集，有效构造模型的能力;
- 模型描述的简洁性和可解释性。模型描述愈简洁、愈容易理解，则愈受欢迎。

## 系列文章

- [Gihutb专栏: 机器学习&深度学习(理论/实践)](https://github.com/media-tm/MTOpenML)
- [CSDN专栏: 机器学习理论与实践](https://blog.csdn.net/column/details/27839.html)
- [CSDN专栏: 深度学习理论与实践](https://blog.csdn.net/column/details/27839.html)

## 参考文献

- [1] 周志华. 机器学习. 清华大学出版社. 2016.
- [2] [日]杉山将. 图解机器学习. 人民邮电出版社. 2015.
- [3] 佩德罗.多明戈斯. 终极算法-机器学习和人工智能如何重塑世界. 中信出版社. 2018.
- [4] 李航. 统计学习方法. 2012.