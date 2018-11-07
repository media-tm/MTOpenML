from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import adam

import pydot

def dump_CNN():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(4,)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4))
    keras.backend.clear_session()
    model.summary()

def dump_LeNet():
    num_classes = 10
    input_shape = (32, 32, 1)
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    keras.backend.clear_session()
    new_model = keras.models.clone_model(model)
    new_model.summary()

def dump_AlexNet():
    num_classes = 10
    input_shape = (32, 32, 1)
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    keras.backend.clear_session()
    model.summary()
    keras.utils.plot_model(model, to_file='alexnet_model.png',show_shapes=True)

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
    help(keras.applications)