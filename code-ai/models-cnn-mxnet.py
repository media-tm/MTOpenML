import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import data as gdata, loss as gloss, nn
import matplotlib.pyplot as plt

def dump_CNN():
    threshold = 'relu' # 'relu' 'sigmoid'
    net = nn.Sequential()  # nn.Sequential is subclass of nn.block
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=6, kernel_size=5, activation=threshold),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation=threshold),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120, activation=threshold),
            nn.Dense(84, activation=threshold),
            nn.Dense(10)
        )
    net.initialize(init=init.Xavier())
    net.summary(mx.nd.ones((1, 1, 28, 28)))

def dump_LeNet():
    threshold = 'relu' # 'relu' 'sigmoid'
    net = nn.Sequential()  # nn.Sequential is subclass of nn.block
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=6, kernel_size=5, activation=threshold),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation=threshold),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Dense(120, activation=threshold),
            nn.Dense(84, activation=threshold),
            nn.Dense(10)
        )
    net.initialize(init=init.Xavier())
    net.summary(mx.nd.ones((1, 1, 28, 28)))

def dump_AlexNet():
    net = nn.Sequential()  # nn.Sequential is subclass of nn.block
    with net.name_scope():
        net.add(
            # 使用较大的 11 x 11 窗口来捕获物体。同时使用步幅 4 来较大减小输出高宽。
            # 这里使用的输入通道数比 LeNet 也要大很多。
            # nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            # nn.MaxPool2D(pool_size=3, strides=2),
            # 减小卷积窗口，使用填充为2来使得输入输出高宽一致。且增大输出通道数。
            nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 连续三个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，
            # 进一步增大了输出通道数。前两个卷积层后不使用池化层来减小输入的高宽。
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 使用比 LeNet 输出大数倍了全连接层。其使用丢弃层来控制复杂度。
            nn.Dense(120, activation="relu"), nn.Dropout(.5),
            nn.Dense(84, activation="relu"), nn.Dropout(.5),
            # 输出层。我们这里使用 FashionMNIST，所以用 10，而不是论文中的 1000。
            nn.Dense(10)
        )
    net.initialize() #init=init.Xavier()
    net.summary(mx.nd.ones((1, 1, 28, 28)))

if __name__ == '__main__':
    print("dump_CNN()")
    dump_CNN()
    print("-")

    print("dump_LeNet()")
    dump_LeNet()
    print("-")

    print("dump_AlexNet()")
    dump_AlexNet()
    print("-")