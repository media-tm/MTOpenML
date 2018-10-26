#
# Python 3.6
#

from tensorflow.python import keras
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

def plot_feature_and_price():
    types_count = 3
    for t,marker,c in zip(range(types_count),'>ox', 'rgb'):
        plt.scatter(data[target==t,x_index],
                    data[target==t,y_index],
                    marker=marker,c=c)
    plt.xlabel(feature_names[x_index])
    plt.ylabel(feature_names[y_index])