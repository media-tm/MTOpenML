# 混合高斯模型(Gaussian Mixture Model)聚类算法

机器学习分为监督学习、无监督学习和半监督学习(强化学习)。无监督学习最常应用的场景是聚类(clustering)和降维(dimension reduction)。聚类算法包括：K均值聚类(K-Means)、层次聚类(Hierarchical Clustering)和混合高斯模型(Gaussian Mixture Model)。降维算法包括：主成因分析(Principal Component Analysis)和线性判别分析(Linear Discriminant Analysis)。

高斯混合模型(Gaussian Mixture Model)简称GMM, 高斯混合模型使用K个高斯分布的结合组成的概率分布模型，并使用了期望最大(Expectation Maximization，简称EM)算法进行训练。其理论基础是：K个高斯分布的结合组成的概率分布模型理论上可以任意地逼近任何连续的概率密分布。

高斯分布(Gaussian distribution)也被称为正态分布(normal distribution)，是一种在自然界大量的存在的、最为常见的分布形式。

## 1 算法原理

混合高斯模型(Gaussian Mixture Model)聚类算法的核心步骤如下:

- 通过观察采样的概率值和模型概率值的接近程度，来判断一个模型是否拟合良好;
- 通过模型来计算数据的期望值，通过更新分布的均值和标准差(参数μ和σ)来让期望值最大化;
- 反复迭代这个过程很多次，直到两个概率值非常接近时;
- 停止更新并完成模型训练。

混合高斯模型(Gaussian Mixture Model)聚类算法的核心优势如下：

- 计算伸缩性: 使用多个高斯分布的组合来刻画数据分布, 计算伸缩性好;
- 参数依赖性: 可调整参数为分布的均值和标准差(参数μ和σ);
- 普适性能力: 描述能力和泛化能力优于K均值聚类(K-Means);
- 抗噪音能力: 由于K均值聚类(K-Means);
- 结果解释性: 模型和结果均具有解释性。

## 2 算法实例

## 3 典型应用

- 用户画像: 在电子商务、新闻客户端、视频客户端通过对用户行为数据的聚类分析，可以准确刻画用户画像。精准的用户画像，对于优化内容推送，内容搜索，广告分发具有重大意义。
- 地理信息: 对于地域性比较强的领域，诸如：汽车保险、个人意外险、房屋租赁和二手房交易等。从用户数据中聚类出此类信息，能够在上述领域充分匹配用户和产品，实现精准营销。

## 参考资料

- [Cmd Markdown 公式指导手册](https://www.zybuluo.com/codeep/note/163962)