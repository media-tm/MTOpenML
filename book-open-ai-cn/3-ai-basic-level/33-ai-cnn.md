# 卷积神经网络(Convolutional Neural Network, CNN)

## 1 CNN的模型

**LeNet5** 诞生于1994年,是最早的卷积神经网络之一,并且推动了深度学习领域的发展。自从1988年开始,在许多次成功的迭代后,这项由 Yann LeCun完成的开拓性成果被命名为**LeNet5**(参见：Gradient-Based Learning Applied to Document Recognition)。

![image](http://static.open-open.com/lib/uploadImg/20160907/20160907100307_377.jpg)

**LeNet5**的架构基于这样的观点：(尤其是)图像的特征分布在整张图像上,以及带有可学习参数的卷积是一种用少量参数在多个位置上提取相似特征的有效方式。在那时候,没有GPU帮助训练,甚至CPU的速度也很慢。因此,能够保存参数以及计算过程是一个关键进展。这和将每个像素用作一个大型多层神经网络的单独输入相反。**LeNet5**阐述了那些像素不应该被使用在第一层,因为图像具有很强的空间相关性,而使用图像中独立的像素作为不同的输入特征则利用不到这些相关性。

## 2 CNN的训练方法

## 3 CNN的学习方法

## 4 LeNet5

**LeNet5**特征能够总结为如下几点：  
1)卷积神经网络使用三个层作为一个系列:卷积,池化,非线性  
2)使用卷积提取空间特征  
3)使用映射到空间均值下采样(subsample)  
4)双曲线(tanh)或S型(sigmoid)形式的非线性  
5)多层神经网络(MLP)作为最后的分类器  
6)层与层之间的稀疏连接矩阵避免大的计算成本  
总体看来,这个网络是最近大量神经网络架构的起点,并且也给这个领域带来了许多灵感。

## 5 CNN知名模型

- LeNet5
- AlexNet
- VGGNet
- ResNet
- GoogLeNet
- XCeption

## 参考文献

1 [CNN(卷积神经网络)概述](http://blog.csdn.net/laingliang/article/details/53073591)  
2 [详解卷积神经网络(CNN)](http://blog.csdn.net/qq_25762497/article/details/51052861)  
3 [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)   
4 [卷积特征提取](http://deeplearning.stanford.edu/wiki/index.php/%E5%8D%B7%E7%A7%AF%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)   
5 [卷积神经网络全面解析](http://www.moonshile.com/post/juan-ji-shen-jing-wang-luo-quan-mian-jie-xi)  
