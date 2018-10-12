# 深度学习-12：深度学习模型的特征与演进(包含20+)

> [CSDN专栏: 机器学习&深度学习(理论/实践)](https://blog.csdn.net/column/details/27839.html)

## 1 神经网络的启蒙模型

### 1.1 Perceptron

### 1.2 Multilayer Perceptron

### 1.3 MLP+SVM

## 2 深度卷积模型

### 2.1 CNN

### 2.2 LeNext

### 2.3 AlexNet

### 2.4 GoogleNet

### 2.5 ResNet

深度残差网络(Deep residual network, ResNet)的提出是CNN图像史上的一件里程碑事件。2014年ImageNet的冠军模型VGG只有19层，2015年ImageNet的冠军模型ResNet多达152层。网络的深度对模型的性能至关重要，当增加网络层数后，网络可以进行更加复杂的特征模式的提取，所以当模型更深时理论上可以取得更好的结果。对于超深层网络模型，具体实践中也会出现严重的网络模型退化问题(Degradation problem), 网络准确度出现饱和，甚至出现下降；也会出现梯度消失或者爆炸的问题。超深层网络模型很难训练，需要很多炼金术的级别的技巧。深度残差网络解决了超深层网络模型很难训练的问题(网络层数提升了一个数量级)。

- 模型特征：(1) ResNet网络是参考了VGG19网络; (2) 机制上引入通过短路机制加入了残差单元。(3) ResNet相比普通网络每两层间增加了短路机制构成残差学习。
- 模型演变: ResNet网络 = VGG19网络 + 残差单元。

## 3 高级深度模型

### 3.1 自编码网络

深度神经网络的网络层数非常多。训练模型时，模型参数的初始化模式很有棘手，选择失当会出现收敛慢、局部最优等问题。自编码网络的提出是为了预训练网络参数，给网络参数一个合适的初值。

- 模型特征：(1)逐层学习策略，相邻的层级快速学习，然后迭代逐层学习。(2) 无监督学习方法，训练数据不需要标签。(3) Hinton提出的初始idea。
- 模型演变: 自编码网络、深度堆栈网络、深度玻尔兹曼机和深度置信网络等。

### 3.2 深度堆栈网络

- 模型特征:(1) 引入堆栈网络机制；(2) 引入自编码网络(逐层学习策略)。
- 模型演变: 深度堆栈网络+自编码网络 = 深度玻尔兹曼机/深度置信网络。

### 3.3 深度融合网络

深度SVM网络
深度PCA网络

### 3.4 深度生成网络

一般的学习模型都是基于一个假设的随机分布，然后通过训练真实数据来拟合出模型。网络模型复杂并且数据集规模也不小，这种方法简直就是凭借天生蛮力解决问题。Goodfellow认为正确使用数据的方式，先对数据集的特征信息有insight之后，再干活。在2014年，Goodfellow等提出生成式对抗网络GAN(Generative adversarial networks)。GAN网络由一个生成器和一个判别器构成。生成器和判别器均可以采用目前研究火热的深度神经网络。

GAN模型特征:

- GAN网络由一个生成器和一个判别器构成;
- GAN网络的生成器捕捉真实数据样本的潜在分布, 并生成新的数据样本;
- GAN网络的判别器是一个二分类器, 判别输入是真实数据还是生成的样本;
- GAN网络的生成器和判别器均可以使用深度学习模型;
- GAN网络的优化过程是极小极大博弈(Minimax game)问题, 优化目标是达到纳什均衡。

GAN模型演变:  

- [Generative Adversarial Networks(GAN)](https://arxiv.org/abs/1406.2661);
- [Conditional Generative Adversarial Nets(CGAN)](https://arxiv.org/pdf/1411.1784);
- [Deep Convolutional Generative Adversarial Nets(DCGAN)](http://arxiv.org/abs/1511.06434);
- [Wasserstein GAN(WGAN)](https://arxiv.org/abs/1701.07875);
- [Improved Training of Wasserstein GANs(WGAN-GP)](https://arxiv.org/abs/1704.00028)
- [Least Squares Generative Adversarial Networks(LSGAN)](https://arxiv.org/abs/1611.04076);
- [Boundary Equilibrium GANs(BEGAN)](https://arxiv.org/abs/1703.10717);
- [Are GANs Created Equal? A Large-Scale Study(Google)](https://arxiv.org/abs/1711.10337)

研究研究这些论文，Github上也有相关实现可以玩一玩。Google在论文《Are GANs Created Equal?》中使用了minimax损失函数和用non-saturating损失函数的GAN，分别简称为MM GAN和NS GAN，对比了WGAN、WGAN GP、LS GAN、DRAGAN、BEGAN等GAN模型变体，发现性能大同小异。这个结论是选择困难症的福音呀。

### 3.5 深度循环神经网络

### 3.6 深度递归神经网络

### 3.7 长短时记忆神经网络

### 3.8 深度强化学习

## 4 移动端的模型

### 4.1 SqueezeNet

### 4.2 MobileNet

### 4.3 ShuffleNet

### 4.4 Xception

## 系列文章

- [CSDN专栏: 机器学习&深度学习(理论/实践)](https://blog.csdn.net/column/details/27839.html)
- [Gihutb专栏: 机器学习&深度学习(理论/实践)](https://github.com/media-tm/MTOpenML)

## 参考文献

- [1] Ian Goodfellow, Yoshua Bengio. [Deep Learning](http://www.deeplearningbook.org/). MIT Press. 2016.
- [2] 焦李成等. 深度学习、优化与识别. 清华大学出版社. 2017.
- [3] 佩德罗·多明戈斯. 终极算法-机器学习和人工智能如何重塑世界. 中信出版社. 2018.
- [4] 雷.库兹韦尔. 人工智能的未来-揭示人类思维的奥秘.  浙江人民出版社. 2016.
