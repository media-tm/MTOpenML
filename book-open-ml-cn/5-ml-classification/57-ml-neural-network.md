# 人工神经网络(Artificial Neural Network)分类算法

> 一起创作,Come on!!! [简练而全面的开源ML&AI电子书](https://github.com/media-tm/MTOpenML)

人工神经网络(Artificial Neural Network)分类算法属于监督学习算法。常用分类算法包括：逻辑回归(Logistic Regression, LR)、K最近邻(k-Nearest Neighbor, KNN)、朴素贝叶斯模型(Naive Bayesian Model, NBM)、隐马尔科夫模型(Hidden Markov Model)、支持向量机(Support Vector Machine)、决策树(Decision Tree)、神经网络(Neural Network)和集成学习(ada-boost)。

人工神经网络(Artificial Neural Network)是模拟神经元的处理信息的数学模型。神经网络包含多个层次，同层之间的神经元相互之间不进行数据通信；相邻层之间的神经元相互联接构成网络，即”神经网络”。数据信息顺着网络正向传播，误差信息逆着网络方向反向传播。

2000年以前，神经网络一直被SVM算法压制，随后出现大量理论创新和应用创新，成为最热门的机器学习方法。人工神经网络(Artificial Neural Network)包括多种模型：BP网络、径向基RBF网络、Hopfield网络、随机神经网络(Boltzmann机)、竞争神经网络(Hamming网络，自组织映射网络)等。神经网络仍普遍存在收敛速度慢、计算量大、训练时间长和不可解释等缺点。

## 1 算法原理

人工神经网络(Artificial Neural Network)分类算法是由多层神经元结构组成，每一层神经元拥有输入和输出。人工神经网络由3个要素组成：拓扑结构、连接方式和学习规则。人工神经网络包含三种层类型:

- 输入层(Input  layer)，输入层神经元可以接受多种类型的数据输入(文字、声音和图像等)。
- 输出层(Output layer)，输出层输出分类或其他决策信息。
- 隐藏层(Hidden layer)，输入层和输出层之间包含多个隐藏层。

人工神经网络(Artificial Neural Network)分类算法的核心步骤如下:

- 数据清洗：数据规范化, 了解数据的基本特征;
- 构建神经网络的拓扑结构
- 确定神经网络的连接方式
- 通过训练样本的校正，得到神经网络的权值矩阵值

人工神经网络(Artificial Neural Network)分类算法的核心优势如下：

- 计算伸缩性: 计算复杂度复杂，基于专用芯片加速可以使用复杂网络结构;
- 参数依赖性: 可调节参数较多;
- 普适性能力: 泛化能力强大；
- 抗噪音能力: 鲁棒性强;
- 结果解释性: 黑箱，不易解释。

## 2 算法实例

## 3 典型应用

人工神经网络(Artificial Neural Network)用于处理语言处理、语音识别、图像识别等领域，表现出优良的性能。

## 参考资料

- [1] 周志华. 机器学习. 清华大学出版社. 2016.
- [2] [日]杉山将. 图解机器学习. 人民邮电出版社. 2015.
- [3] 佩德罗·多明戈斯. 终极算法-机器学习和人工智能如何重塑世界. 中信出版社. 2018.
- [4] 李航. 统计学习方法. 2012.