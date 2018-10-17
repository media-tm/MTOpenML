# Deep CNN(AlexNet)

AlexNet跟LeNet结构类似，但使用了更多的卷积层和更大的参数空间来拟合大规模数据集ImageNet。它是浅层神经网络和深度神经网络的分界线。虽然看上去AlexNet的实现比LeNet也就就多了几行而已。但这个观念上的转变和真正优秀实验结果的产生，学术界整整花了20年。

## 1 AlexNet的演进

## 2 AlexNet的网络结构

## 3 AlexNet的创新

- 引入了新的激活函数ReLu
- 通过丢弃法来控制全连接层的模型复杂度。
- 引入了大量的图片增广，例如翻转、裁剪和颜色变化，进一步扩大数据集来减小过拟合。

## 4 AlexNet的代码实现

## 5 参考文献

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).
[5] [AlexNet原理及Tensorflow实现](https://blog.csdn.net/taoyanqi8932/article/details/71081390)