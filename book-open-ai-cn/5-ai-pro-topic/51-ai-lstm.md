# Long Short-term Memory Networks(LSTM)

Long Short Term网络一般叫做 LSTM，是一种 RNN 特殊的类型，可以学习长期依赖信息。LSTM 由 Hochreiter & Schmidhuber (1997) 提出，并在近期被 Alex Graves 进行了改良和推广。STM 通过刻意的设计来避免长期依赖问题。记住长期的信息在实践中是 LSTM 的默认行为，而非需要付出很大代价才能获得的能力！

人类的每一次思考并非总是重新开始的，思考过程中总是伴随着很多阶段，下一步的思考总是承接上几步的思考，迭代堆叠的产生新的思考结果。传统的神经网络并不能做到这点，看起来也像是一种巨大的弊端。RNN 解决了这个问题。RNN 是包含循环的网络，允许信息的持久化。

## 1 LSTM网络的演进

## 2 LSTM网络的结构

## 3 LSTM网络的创新

- 解决梯度消失问题
- 解决训练数据不足的问题

## 4 LSTM网络的实现

## 参考文献

[x] Sepp Hochreiter; Jürgen Schmidhuber (1997). "Long short-term memory". Neural Computation. 9 (8): 1735–1780. doi:10.1162/neco.1997.9.8.1735. PMID 9377276.
[x] Graves, Alex; Mohamed, Abdel-rahman; Hinton, Geoffrey (2013-03-22). "Speech Recognition with Deep Recurrent Neural Networks". arXiv:1303.5778
[x] Tax, N.; Verenich, I.; La Rosa, M.; Dumas, M. (2017). "Predictive Business Process Monitoring with LSTM neural networks". arXiv:1612.02130 