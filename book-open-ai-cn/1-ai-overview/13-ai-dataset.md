# 开源AI数据集

## 1 机器视觉图形图像数据集

### 1.1 Mnist

下载地址：http://yann.lecun.com/exdb/mnist/index.html  
深度学习领域的“Hello World!”，入门必备！MNIST是一个手写数字数据库，它有60000个训练样本集和10000个测试样本集，每个样本图像的宽高为28*28。

### 1.2 ImageNet

下载地址：http://www.image-net.org/about-stats  
MNIST将初学者领进了深度学习领域，而Imagenet数据集对深度学习的浪潮起了巨大的推动作用。深度学习领域大牛Hinton在2012年发表的论文《ImageNet Classification with Deep Convolutional Neural Networks》在计算机视觉领域带来了一场“革命”，此论文的工作正是基于Imagenet数据集。

Imagenet数据集有1400多万幅图片，涵盖2万多个类别；其中有超过百万的图片有明确的类别标注和图像中物体位置的标注，具体信息如下：

- Total number of non-empty synsets: 21841  
- Total number of images: 14,197,122  
- Number of images with bounding box annotations: 1,034,908  
- Number of synsets with SIFT features: 1000  
- Number of images with SIFT features: 1.2 million  

### 1.3 COCO

下载地址：http://mscoco.org/  
COCO(Common Objects in Context)是一个新的图像识别、分割和图像语义数据集，它有如下特点：

- Object segmentation  
- Recognition in Context  
- Multiple objects per image  
- More than 300,000 images  
- More than 2 Million instances  
- 80 object categories  
- 5 captions per image  
- Keypoints on 100,000 people  

COCO数据集由微软赞助，其对于图像的标注信息不仅有类别、位置信息，还有对图像的语义文本描述，COCO数据集的开源使得近两三年来图像分割语义理解取得了巨大的进展，也几乎成为了图像语义理解算法性能评价的“标准”数据集。

### 1.4 CIFAR

下载地址：http://www.cs.toronto.edu/~kriz/cifar.html  
CIFAR-10包含10个类别，50,000个训练图像，彩色图像大小：32x32，10,000个测试图像。CIFAR-100与CIFAR-10类似，包含100个类，每类有600张图片，其中500张用于训练，100张用于测试；这100个类分组成20个超类。图像类别均有明确标注。CIFAR对于图像分类算法测试来说是一个非常不错的中小规模数据集。

### 1.5 Open Image

下载地址：https://github.com/openimages/dataset  
Open Image是一个包含~900万张图像URL的数据集，里面的图片通过标签注释被分为6000多类。该数据集中的标签要比ImageNet（1000类）包含更真实生活的实体存在，它足够让我们从头开始训练深度神经网络。

## 2 视频数据集

[Youtube-8M](https://research.google.com/youtube8m/)为谷歌开源的视频数据集，视频来自youtube，共计8百万个视频，总时长50万小时，4800类。为了保证标签视频数据库的稳定性和质量，谷歌只采用浏览量超过1000的公共视频资源。为了让受计算机资源所限的研究者和学生也可以用上这一数据库，谷歌对视频进行了预处理，并提取了帧级别的特征，提取的特征被压缩到可以放到一个硬盘中（小于1.5T）。

[CDVL(TheConsumer Digital Video Library)](http://www.cdvl.org/) 消费者数字视频库对外提供高质量的源视频序列，可供研究和开发免费使用。CDVL还托管了几个视频质量数据集，包括五个VQEG HD Phase I数据集，BVI-HD，CCRIQ，its4s和T1A1。

[LIVE database](http://live.ece.utexas.edu/research/Quality/) 出自德克萨斯大学的图像&视频工程实验室。该实验室的视觉科学家和视频工程师对图片和视频质量进行大规模主观和客观研究，对相关数据库做了严格的视觉检测/筛选。该数据库包含15+细分的自数据库。该数据库还包含若干视频质量评估的背景知识。

IVC数据库包含图像质量评价和视频质量评价数据库。IVC数据库由法国南特大学(Université de Nantes)的南特通信与网络研究所主持构建和维护。[南特通信与网络研究所](http://ivc.univ-nantes.fr/en/)在图像&视频质量评价、离散信息表示、人类世界感知、机器学习和模式识别、网络和系统等方面具有深刻而广泛的研究。

## 3 音频数据集

谷歌发布的大规模一品数据集，[AudioSet](https://github.com/audioset/ontology) 包括 632 个音频事件类的扩展类目和从YouTube视频绘制的 2084320 个人类标记的10秒声音剪辑的集合。类目被指定为事件类别的分层图，覆盖广泛的人类和动物声音，乐器和风格以及常见的日常环境声音。

[2000 HUB5 English Evaluation Transcripts](https://catalog.ldc.upenn.edu/LDC2002T43)由NIST（国家标准与技术研究院）2000年发起的HUB5评估中使用的40个英语电话对话的成绩单组成，其仅包含英语的语音数据集，百度最近的论文《深度语音：扩展端对端语音识别》使用的是这个数据集。

[TED-LIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus)是TED Talk的音频数据集，包含1495个录音和音频会议、159848条发音词典和部分WMT12公开的语料库。

## 4 综合数据集

- [Amazon公开数据集](http://aws.amazon.com/datasets)
- [100多个有趣的数据集-CSDN](http://www.csdn.net/article/2014-06-06/2820111-100-Interesting-Data-Sets-for-Statistics)