# Inception Architecture(Inception)

在2014年的Imagenet竞赛中，一个名叫GoogLeNet [1]的网络结构大放光彩。它虽然在名字上是向LeNet致敬，但在网络结构上已经很难看到LeNet的影子。GoogLeNet吸收了NiN的网络嵌套网络的想法，并在此基础上做了很大的改进。

Xception是google继Inception后提出的对Inception v3的另一种改进，主要是采用depthwise separable convolution来替换原来Inception v3中的卷积操作。

要介绍Xception的话，需要先从Inception讲起，Inception v3的结构图如下Figure1。当时提出Inception的初衷可以认为是：特征的提取和传递可以通过1*1卷积，3*3卷积，5*5卷积，pooling等，到底哪种才是最好的提取特征方式呢？Inception结构将这个疑问留给网络自己训练，也就是将一个输入同时输给这几种提取特征方式，然后做concat。Inception v3和Inception v1（googleNet）对比主要是将5*5卷积换成两个3*3卷积层的叠加。

## 1 Inception的演进

## 2 Inception的网络结构

## 3 Inception的创新

- 础卷积块定义为Inception
- 大量使用NIN(network in network)

## 4 Inception的代码实现

## 参考文献

[1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., & Anguelov, D. & Rabinovich, A.(2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
[2] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.
[3] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818-2826).
[4] Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017, February). Inception-v4, inception-resnet and the impact of residual connections on learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 4, p. 12).
[5] F. Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions,” 1610.02357, https://arxiv.org/pdf/1610.02357.pdf.