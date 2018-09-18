# Deep Residual Network(ResNet)

对于深层网络来说，还有一个问题困扰着训练：在进行梯度反传计算时，我们从误差函数（顶部）开始，朝着输入数据方向（底部）逐层计算梯度。当我们将层串联在一起时，根据链式法则每层的梯度会被乘在一起，这样便导致梯度数值以指数衰减。最后，在靠近底部的层只得到很小的梯度，对应的权重更新量也变小，使得他们的收敛缓慢。

ResNet [1] 成功地通过增加跨层的数据线路来允许梯度快速地到达底部层，从而避免这一情况。这一节我们将介绍ResNet的工作原理。

## 1 ResNet的演进

## 2 ResNet的网络结构

## 3 ResNet的创新

- 使用超深层级的网络模型

## 4 ResNet的代码实现

## 参考文献
[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings in deep residual networks. In European Conference on Computer Vision (pp. 630-645). Springer, Cham.