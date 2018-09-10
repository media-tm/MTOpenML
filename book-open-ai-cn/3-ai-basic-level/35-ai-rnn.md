# 3.4 循环神经网络(Recurrent Neural Network, RNN)

在图像分类和目标识别领域，基于前馈神经网络的深度学习模型表现优异，但是在语音识别和自然语音处理领域深度学习模型水土不服，时间序列数据存在时间关联性和整体逻辑特性。深度学习模型无法利用历史数据的时间依赖关系，来分析数据特征，故而无法处理时间序列数据。计算机科学家借鉴大脑处理时序数据的模式，改造了深度学习模型，提出了循环神经网络模型。循环神经网络模型在隐层引入了定向循环，它能更好的表征高纬度信息的整体逻辑性。

## 1 RNN简史
- 1982年，约翰·霍普菲尔德提出Hopfiled网络，网络内部有反馈连接，能够处理信号中的时间依赖性；
- 1986年，Michael Jordan 在神经网络中引入循环连接；
- 1990年，Jeffrey Elman 正式提出了RNN模型，RNN具备有限短期记忆；(利用反向传播和梯度下降算法过程中遭受到了严重的梯度消失问题)
- 1997年，Sepp Hochreiter提出长短期记忆(Long short-term memory)网络模型；
- 2003年，Yoshua Bengio基于RNN的N元统计模型，解决了分词特征表征和维度魔咒问题。
- 2010年以后，循环神经网络(RNN)/卷积神经网络（CNN）/深度信念网络（DBN）成为深度学习的三个模型。并诞生了很多智能语音应用(SIRI,Alexa...)。

## 2 RNN的生物机理

时间认知和目标导向在人脑处理语音识别和自然语音处理活动时显然非常重要。“书读百遍，其义自见”，“一回生二回熟”，“失败是成功母”等俗语均能发现时间认知和目标导向在人脑处理信息时的作用。历史信息经过强化叠加，逐渐沉淀下来，最终成为我们的经验知识，经验知识和上下文场景在某些场合可能比真实输入数据还要重要。RNN通过使用带有自反馈的神经元，能够处理理论上任意长度的（存在时间关联性的）序列数据。相比于传统的前馈神经网络，它更符合生物神经元的连接方式，更符合人类大脑处理信息的工作模式。

## 3 RNN的模型描述

时间序列数据存在时间关联性和整体逻辑特性，传统神经网络信息是单向传播的(误差反向传播不改变单向性)，循环神经网络(RNN)将输出层的结果再次输入到隐藏层，以其能够发现时间序列数据的时间关联性和整体逻辑特性等高维度信息。1990年，Jeffrey Elman正式提出了RNN模型，RNN具备有限短期记忆；1997年，Sepp Hochreiter提出长短期记忆(LSTM)网络模型。

### 3.1 经典循环神经网络(RNN)

和经典神经网络一样，循环神经网络(RNN)包含输入层，隐藏层和输出层；隐藏层分配了若干个权重矩阵，并利用优化函数(损失函数)按照一定优化条件定向优化，限定实际输出与目标输出之间的误差，最终计算出网络的权重矩阵。循环神经网络(RNN)引入了两个创新，(1) 即时输入和历史输入均分配权重，并通过优化函数确认最终的权重。(2) 引入了时间反向传播(BackPropagation Through Time，简称BPTT）。

#### 3.1.1 网络优化

循环神经网络(RNN)继承了常规的梯度下降算法优化网络模型，常用随机梯度下降算法等均可用于优化循环神经网络(RNN)模型。梯度下降算法通过放大多维误差或代价函数的局部最小值来打破维数灾难。关于梯度下降算法可参考：

#### 通过时间的反向传播

经典神经网络使用著名的BP算法反向逐层回归优化网络模型，利用损失函数的偏导调整每个单元的权重。循环神经网络(RNN)使用新版本机制即时间的反向传播(BPTT),引入了记忆单元，历史时刻T-1/T-2的数据也参与计算，用以关注时间序列数据的 时间关联性和整体逻辑特性。当然记忆模块可以有多种变体，比如长时记忆、短时记忆和工作记忆，不一而足。

#### 梯度消失问题
初代的循环神经网络(RNN)获取了初步的成功，但是梯度消失问题像梦魇一样让其陷入研究的低谷。梯度可以视为斜率，梯度值就是优化指示器；梯度值越大，网络模型很快速收敛到最佳状态；梯度值平坦或微弱，网络优化就失去了优化目标，网络模型无法收敛到最佳状态，只能处于随机漫步状态。
 
### 3.2 新一代LSTM

理论研究有时曲线前进的，进10步退两步，曲折中前行。1997年，Sepp Hochreiter提出长短期记忆(LSTM)网络模型从理论和时间上解决了循环神经网络(RNN)的梯度消失问题。从此长短期记忆(LSTM)网络模型引发了新的研究热潮。长短期记忆(LSTM)网络模型的核心思想思想是引入了长短时记忆单元，该单元与标准的RNN的标准短期记忆单元相比，具有一些新特性。长短时记忆单元能够保留、删除、转换和控制其存储数据的流入和流出。长短时记忆单元可以长时间保存重要的错误信息，以使梯度相对陡峭，从而网络的训练时间相对较短，这种机制解决了梯度消失的问题。

### 3.3 RNN的优势
- 循环神经网络(RNN)处理时间序列数据具有先天优势；
- 循环神经网络(RNN)通过反向传播和梯度下降算法达到了纠正错误的能力，但未解决梯度消失问题；
- 直到1997年，循环神经网络(RNN)引入了一个基于LSTM的架构后，梯度消失问题得以解决；
- LSTM的架构中的单元相当于一个模拟计算机，显著提高了网络精度。

### 4 RNN代码示例
- [MxNet的RNN示例]()
- [Tensorflow的RNN示例]()

### 5 扩展思考
5.1 LTSM的原理，LTSM对RNN的主要改进是什么？
5.2 请列举基于RNN/LTSM的流行产品或应用?
5.3 RNN/LTSM可以预测股票等金融数据吗？

## 参考文献 （TODO）
[1] Hopfield J J. Neural networks and physical systems with emergent collective computational abilities [J]. Proceedings of the National Academy of Sciences of the United States of America, 1982, 79(8):2554.
[2] Jordan, M. (1986). Serial order: A parallel distributed processing approach. Institute for Cognitive Science Report 8604. University of California, San Diego.
[3] Elman J L. Finding structure in time[J]. Cognitive Science, 1990, 14(2):179-211.
[4] Sepp Hochreiter; Jürgen Schmidhuber (1997). "Long short-term memory". Neural Computation. 9 (8): 1735–1780. PMID 9377276. doi:10.1162/neco.1997.9.8.1735.
[5] Bengio Y, Vincent P, Janvin C. A neural probabilistic language model[J]. Journal of Machine Learning Research, 2003, 3(6):1137-1155.
[6] A. Graves. Supervised Sequence Labelling with Recurrent Neural Networks. Textbook, Studies in Computational Intelligence, Springer, 2012.
[7] Jiang X, Shen S, Cadwell C R, et al. Principles of connectivity among morphologically defined cell types in adult neocortex.[J]. Science, 2015, 350(6264):aac9462.


