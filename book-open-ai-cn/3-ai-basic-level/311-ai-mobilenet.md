# Efficient Convolutional Network for Mobile(MobileNet)

Google公司提出了一类称为MobileNets的高效模型适用于移动和嵌入式视觉应用。MobileNets基于使用depthwise的简化架构可分离的卷积，重量轻神经网络。 我们介绍两个简单的全局超参数有效地在延迟和延迟之间进行权衡准确性。 这些超参数允许模型构建器为他们的应用选择合适大小的模型关于问题的限制。 我们提供广泛的资源和准确性权衡和展示的实验与其他热门机型相比，性能强劲ImageNet分类。 然后我们证明了有效性MobileNets的广泛应用和用例包括物体检测，细粒度分类，面部属性和大规模地理定位。

## 1 MobileNet的演进

## 2 MobileNet的网络结构

## 3 MobileNet的创新

- MobileNet V1 的准确率不错，速度很快
- MobileNet V2 引入了Linear Bottleneck 和 Inverted Residual Blocks
- 采用 Depth-wise (DW) 卷积搭配 Point-wise (PW) 卷积的方式来提特征

## 4 MobileNet的代码实现

## 参考文献

A. G. Howard, M. Zhu, B. Chen et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” 1704.04861, https://arxiv.org/pdf/1704.04861.pdf.