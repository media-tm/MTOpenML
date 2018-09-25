# K最近邻(k-Nearest Neighbor-KNN)分类算法

> 一起创作,Come on!!! [简练而全面的开源ML&AI电子书](https://github.com/media-tm/MTOpenML)

K最近邻(k-Nearest Neighbor-KNN)分类算法属于监督学习算法。常用分类算法包括：逻辑回归(Logistic Regression, LR)、K最近邻(k-Nearest Neighbor, KNN)、朴素贝叶斯模型(Naive Bayesian Model, NBM)、隐马尔科夫模型(Hidden Markov Model)、支持向量机(Support Vector Machine)、决策树(Decision Tree)、神经网络(Neural Network)和集成学习(ada-boost)。

1968年，Cover 和 Hart 提出 K最近邻(k-Nearest Neighbor-KNN)分类算法。其核心想法非常简单明了，确定一个临近度的度量, 相似性越高，相异性越低的数据样本，可以认为是同一个数据类别。

## 1 算法原理

K最近邻(k-Nearest Neighbor-KNN)分类算法是采用测量不同数据特征值之间的距离方法进行分类。有几种临近度的度量可以用于K最近邻分类算法：欧几里得距离、二元数据的相似性度量和余弦相似度等。欧几里得距离是最常用的距离公式。距离对特征都是区间或比率的对象非常有效。两个仅包含二元属性的对象之间的相似性度量也称为相似系数(similarity coefficient)。余弦相似度(cosine similarity)就是文档相似性最常用的度量之一。如果x和y是两个文档向量:  
$cos(x,y)=(x*y)/(||x||*||y||)$

K最近邻(k-Nearest Neighbor-KNN)分类算法的核心步骤如下:

- 数据清洗：数据规范化，例如年龄不超过150等;
- 确定临近度的度量，并计算临近度(诸如: 数据集中的点与当前点之间的距离);
- 按照临近度递增次序排序;
- 选取与当前点距离最小的k个点;
- 确定前k个点所在类别的出现频率;
- 返回前k个点出现频率最高的类别作为当前点的预测分类。

K最近邻(k-Nearest Neighbor-KNN)分类算法的核心优势如下：

- 计算伸缩性: 计算复杂度可控;
- 参数依赖性: 可调节参数较少。k值取得过小，容易受噪点的影响; 而k值取得过大，分类不明确;
- 普适性能力: 适用于离散型数据集；
- 抗噪音能力: k值取得过小，容易受噪点的影响;
- 结果解释性: 理论简单解释性好。

## 2 算法实例

## 3 典型应用

K最近邻(k-Nearest Neighbor-KNN)分类算法广泛应用于字符识别、文本分类、图像识别等领域。

## 参考资料

- [1] 周志华. 机器学习. 清华大学出版社. 2016.
- [2] [日]杉山将. 图解机器学习. 人民邮电出版社. 2015.
- [3] 佩德罗·多明戈斯. 终极算法-机器学习和人工智能如何重塑世界. 中信出版社. 2018.
- [4] 李航. 统计学习方法. 2012.
- [5] [分类:K Nearest Neighbour)最近邻算法](https://www.cnblogs.com/fushengweixie/p/8196371.html)