# 主成因分析(Principal Component Analysis)降维算法

机器学习分为监督学习、无监督学习和半监督学习(强化学习)。无监督学习最常应用的场景是聚类(clustering)和降维(dimension reduction)。聚类算法包括：K均值聚类(K-Means)、层次聚类(Hierarchical Clustering)和混合高斯模型(Gaussian Mixture Model)。降维算法包括：主成因分析(Principal Component Analysis)和线性判别分析(Linear Discriminant Analysis)。

主成因分析(Principal Component Analysis: PCA)是使用最广的降维方法。PCA顾名思义，就是找出数据里最主要的方面，用数据里最主要的方面来代替原始数据。

## 1 算法原理

主成因分析(Principal Component Analysis)降维算法是一种非监督学习的降维方法。PCA算法利用特征值分解的思想和事件，将高维数据压缩或去噪成低维数据。PCA算法由于某些固有缺陷，出现了很多PCA算法变种: 解决非线性降维的PCA方法、解决内存限制的增量PCA方法和解决稀疏数据降维的PCA方法。

假设我们的数据集是n维的，数据集中有m个数据{X1, X2, ..., Xm}。我们希望将这个数据集从n维压缩到n/2维。这个压缩明显是有损的压缩，我们唯一能做的是保证变换后的n/2维数据集能够最大限度的表征原始的n维数据集。这里我们采用的核心技巧就是特征值分解，保证泛化性能。

主成因分析(Principal Component Analysis)降维算法的核心步骤如下:

- 对所有的样本进行中心化;
- 计算样本的协方差矩阵T;
- 对协方差矩阵T进行特征值分解;
- 取出最大的n'个特征值对应的特征向量(w1,w2,...,wn′), 计算标准化特征向量矩阵W;
- 取出原始高维度样本x(i), 进行降维计算操作，获得低维度样本: z(i)=WTx(i);
- 输出降维后的样本集D′=(z(1),z(2),...,z(m))。

主成因分析(Principal Component Analysis)降维算法的核心优势如下：

- 计算伸缩性: 计算方法简单，主要运算是特征值分解，易于实现;
- 参数依赖性: 以方差衡量信息量，不受数据集以外的因素影响;
- 普适性能力: 特征值分解时各主成分之间正交，泛化能力较好;
- 抗噪音能力: 泛化能力较好，抗噪音能力强;
- 结果解释性: 特征维度的含义可能具有模糊性，解释性降低。

## 2 算法实例

## 3 典型应用

主成因分析(Principal Component Analysis: PCA)在数据压缩消除冗余和数据噪音消除等领域都有广泛的应用。

## 参考资料

- [Cmd Markdown 公式指导手册](https://www.zybuluo.com/codeep/note/163962)