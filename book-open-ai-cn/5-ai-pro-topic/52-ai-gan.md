# Generative Adversarial Networks(GAN)

一个通过对抗过程来估计生成模型的新框架。其中我们同时训练两个模型：一个是生成分布模型G,其作用为产生新的数据，另一个为判别模型D，用于进行估计，估计模型G产生的数据是否来自训练数据的概率。这个框架可以对应于一个极小的双人游戏。随着函数G不断恢复训练数据分布，此时任意位置G和D的可能性等于1/2。在G和D由多层感知器定义的情况下，整个系统可以用反向传播来训练。在训练或生成样本期间，不需要任何马尔可夫链或展开的近似推理网络。实验通过对生成的样本进行定性和定量评估来证明框架的潜力。

## 1 GAN网络的演进

## 2 GAN网络的结构

## 3 GAN网络的创新

- 解决梯度消失问题
- 解决训练数据不足的问题

## 4 GAN网络的实现

## 参考文献

[x] https://www.zhihu.com/question/52602529
[x] https://zhuanlan.zhihu.com/p/31028821
[x] https://blog.csdn.net/leceall/article/details/78551169
[x] Goodfellow, Ian, et al. "Generative adversarial nets." 
    Advances in Neural Information Processing Systems. 2014.
[x] Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets."arXiv preprint 
    arXiv:1411.1784, https://arxiv.org/abs/1411.1784
[x] Arjovsky M, Chintala S, Bottou L. Wasserstein GAN[J]. 
    arXiv:1701.07875, https://arxiv.org/abs/1701.07875
[x] Wang, Jun, et al. "IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models." 
    arXiv:1705.10513, https://arxiv.org/abs/1705.10513
[x] Chen, Xi, et al. "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets." 
    arXiv:1606.03657, https://arxiv.org/abs/1606.03657
[x] Radford, Alec, Luke Metz, "Unsupervised representation learning with deep convolutional generative adversarial networks." 
    arXiv:1511.06434, https://arxiv.org/abs/1511.06434