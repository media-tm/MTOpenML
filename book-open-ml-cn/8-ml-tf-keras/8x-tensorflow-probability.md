# TensorFlow Probability概率编程工具箱快速上手

在2018年TensorFlow开发者峰会上，Google宣布了TensorFlow Probability项目。该项目是一种概率编程工具箱，用于机器学习研究人员和其他从业人员快速可靠地利用最先进硬件构建复杂模型。TensorFlow Probability 适用于以下需求：

- 希望建立一个生成数据模型，推理其隐藏进程。
- 需要量化预测中的不确定性，而不是预测单个值。
- 训练集具有大量相对于数据点数量的特征。
- 结构化数据（例如，使用分组，空间，图表或语言语义）并且你想获取其中重要信息的结构。

TensorFlow Probability为你提供解决上述这些问题的工具，此外，它还继承了TensorFlow的优势，如自动差异化，以及跨多种平台(CPU，GPU和TPU)扩展性能的能力。

## 1. 什么是TensorFlow Probability？

我们这次发布的机器学习工具为TensorFlow生态系统中的概率推理和统计分析提供了模块化抽象。

TensorFlow概率的概述。概率编程工具箱为从数据科学家和统计人员到所有TensorFlow用户的用户提供了好处。

### 1.1 第0层 TensorFlow的数值运算

TensorFlow的数值运算。特别是，LinearOperator类实现了无矩阵计算，可以利用特殊结构(对角线，低秩矩阵等)进行高效计算。它由TensorFlow Probability团队构建和维护，现在是TF中tf.linalg核心的一部分。

### 1.2 第1层：统计构建块

- 分布(tfp.distributions，tf.distributions)：具有批处理和广播语义的大量概率分布和相关统计。请参阅“分发教程”。
- Bijectors(tfp.bijectors)：随机变量的可逆和可组合变换。投影仪提供了丰富的变换分布，从对数正态分布等经典实例到掩盖的自回归流等复杂的深度学习模型。

### 1.3 第2层：模型构建

- Edward2(tfp.edward2)：一种概率编程语言，用于将灵活的概率模型指定为程序。
- 概率层(tfp.layers)：神经网络层，它们代表的功能具有不确定性，扩展了TensorFlow图层。
- 可训练分布(tfp.trainable_distributions)：由单个Tensor参数化的概率分布，可以轻松构建输出概率分布的神经网络。

### 1.4 第3层：概率推理

- 马尔可夫链蒙特卡罗(tfp.mcmc)：通过采样逼近积分的算法。包括Hamiltonian Monte Carlo，随机漫步Metropolis-Hastings，以及构建自定义转换内核的能力。
- 变分推理(tfp.vi)：通过优化逼近积分的算法。
- 优化器(tfp.optimizer)：随机优化方法，扩展TensorFlow优化器。包括随机梯度Langevin动力学。
- 蒙特卡洛(tfp.monte_carlo)：计算蒙特卡洛期望的工具。

### 1.5 第4层：预制模型和推理(用户自定义实现)

- 贝叶斯结构时间序列：用于拟合时间序列模型的高级接口。
- 广义线性混合模型：用于拟合混合效应回归模型的高级界面。

## 2 几个实例带你飞

### 2.1 Edward2打造的线性混合效应模型

线性混合效应模型是对数据中结构化关系进行建模的简单方法，也可以称为分级线性模型，它分享各组数据点之间的统计强度，以便改进对任何单个数据点的推论。

作为演示，请考虑R中流行的lme4包中的InstEval数据集，其中包含大学课程及其评估评级。使用TensorFlow Probability，我们将模型指定为Edward2概率程序（tfp.edward2），它扩展了Edward。下面的程序根据其生成过程来确定模型:

``` python
import tensorflow as tf
from tensorflow_probability import edward2 as ed
def model(features):
  # Set up fixed effects and other parameters.
  intercept = tf.get_variable("intercept", [])
  service_effects = tf.get_variable("service_effects", [])
  student_stddev_unconstrained = tf.get_variable(
      "student_stddev_pre", [])
  instructor_stddev_unconstrained = tf.get_variable(
      "instructor_stddev_pre", [])
  # Set up random effects.
  student_effects = ed.MultivariateNormalDiag(
      loc=tf.zeros(num_students),
      scale_identity_multiplier=tf.exp(
          student_stddev_unconstrained),
      name="student_effects")
  instructor_effects = ed.MultivariateNormalDiag(
      loc=tf.zeros(num_instructors),
      scale_identity_multiplier=tf.exp(
          instructor_stddev_unconstrained),
      name="instructor_effects")
  # Set up likelihood given fixed and random effects.
  ratings = ed.Normal(
      loc=(service_effects * features["service"] +
           tf.gather(student_effects, features["students"]) +
           tf.gather(instructor_effects, features["instructors"]) +
           intercept),
      scale=1.,
      name="ratings")
return ratings
```

该模型将“服务”“学生”和“教师”的特征字典作为输入，它们是每个元素描述单个课程的向量。该模型回归这些输入，假设潜在的随机变量，并返回课程评估评分的分布。在此输出上运行的TensorFlow会话将返回一代评级。

查看“线性混合效应模型”教程，详细了解如何使用tfp.mcmc.HamiltonianMonteCarlo算法训练模型，以及如何使用后预测来探索和解释模型。

### 2.2 高斯Copulas与TFP Bijectors

Copulas是一个多元概率分布，其中每个变量的边缘概率分布是均匀的。要构建使用TFP内在函数的copula，可以使用Bijectors和TransformedDistribution，这些抽象可以轻松创建复杂的分布，例如：

``` python
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.distributions.bijectors
# Example: Log-Normal Distribution
log_normal = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.Exp())
# Example: Kumaraswamy Distribution
Kumaraswamy = tfd.TransformedDistribution(
    distribution=tfd.Uniform(low=0., high=1.),
    bijector=tfb.Kumaraswamy(
        concentration1=2.,
        concentration0=2.))
# Example: Masked Autoregressive Flow
# https://arxiv.org/abs/1705.07057
shift_and_log_scale_fn = tfb.masked_autoregressive_default_template(
    hidden_layers=[512, 512],
    event_shape=[28*28])
maf = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),     
    bijector=tfb.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=shift_and_log_scale_fn))
```

该“高斯系 Copula”创建了一些自定义Bijectors，然后展示了如何轻松地建立多个不同的Copula函数。有关分配的更多背景信息，请参阅“了解张量流量分布形状”。它介绍了如何管理抽样，批量训练和建模事件的形状。

### 2.3 变分自动编码器

变分自动编码器是一种机器学习模型，其使用一个学习系统来表示一些低维空间中的数据，并且使用第二学习系统来将低维表示还原为本来是输入的。TFP支持自动分化，很容易实现黑盒变换推理。

``` python
import tensorflow as tf
import tensorflow_probability as tfp
# Assumes user supplies `likelihood`, `prior`, `surrogate_posterior`
# functions and that each returns a 
# tf.distribution.Distribution-like object.
elbo_loss = tfp.vi.monte_carlo_csiszar_f_divergence(
    f=tfp.vi.kl_reverse,  # Equivalent to "Evidence Lower BOund"
    p_log_prob=lambda z: likelihood(z).log_prob(x) + prior().log_prob(z),
    q=surrogate_posterior(x),
    num_draws=1)
train = tf.train.AdamOptimizer(
    learning_rate=0.01).minimize(elbo_loss)
```

### 2.4 具有TFP概率层的贝叶斯神经网络

贝叶斯神经网络是一个神经网络，它的权重和偏差具有先验分布。它通过这些先验提供了改进的不确定性。贝叶斯神经网络也可以解释为神经网络的无限集合：分配给每个神经网络配置的概率是根据先前的。

作为一个小例子，我们使用了具有特征(形状为32 x 32 x 3的图像)和标签（值为0到9）的CIFAR-10数据集。为了拟合神经网络，我们将使用变分推理，这是一套方法来逼近神经网络在权重和偏差上的后验分布。也就是说，我们在TensorFlow Probabilistic Layers模块中使用最近发布的Flipout估计器tfp.layers。

``` python
class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tfp.layers.DenseFlipout(units=10)
    self.dense2 = tfp.layers.DenseFlipout(units=10)
  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    # reuse variables from dense2 layer
    result = self.dense2(result)  
    return result
model = MNISTModel()
```

### 2.5 其他示例

请参阅tensorflow_probability/examples/目录, tensorflow-probability项目本身自带大量示例：

#### 初级案例

- 线性混合效应模型：用于在示例之间共享统计强度的分层线性模型。
- 八所学校：可交换治疗效果的分级正常模型。
- 分层线性模型：在TensorFlow概率，R和Stan之间进行比较的分层线性模型。
- 贝叶斯高斯混合模型：使用概率生成模型进行聚类。
- 概率主成分分析：使用潜在变量减少维数。
- 高斯Copulas：捕获随机变量依赖性的概率分布。
- TensorFlow发行版：一个温和的介绍。TensorFlow发行版简介。
- TensorFlow分布形状：如何区分任意形状的概率计算的样本，批次和事件。

#### 高级案例

- 协方差估计：用户在应用TensorFlow概率估计协方差时的案例研究。
- 变分自动编码器。使用潜在代码和变分推理进行表示学习。
- 解缠的序贯变分自动编码器具有变分推理的序列的解缠结表示学习。
- 语法变分自动编码器。表示在无上下文语法中学习制作。
- Latent Dirichlet Allocation(Edward2版本)。用于捕获文档中主题的混合成员资格建模。
- 深度指数家庭。用于发现主题层次结构的深层稀疏生成模型。
- 贝叶斯神经网络。神经网络的权重不确定。
- 贝叶斯Logistic回归。二元分类的贝叶斯推断。

## 3 安装使用

使用PIP安装tensorflow-probability，开始您的概率机器学习之旅:

``` python
pip install --user --upgrade tensorflow-probability
pip install --user --upgrade tensorflow-probability-gpu
python tensorflow_probability/examples/logistic_regression.py --help
```

对于所有的代码和细节，请查看 github.com/tensorflow/probability。

## 4 系列文章

- [深度学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83040730)
- [机器学习原理与实践(开源图书)-总目录](https://blog.csdn.net/shareviews/article/details/83030331)
- [Github: 机器学习&深度学习理论与实践(开源图书)](https://github.com/media-tm/MTOpenML)