#
# Python 3.6
#

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

iris_ds = load_iris()

print(iris_ds.keys())
print("iris_ds.keys()  = {0}".format(iris_ds.keys()))

data = iris_ds['data']
print("iris_ds.data.shape  = {0}".format(data.shape))

feature_names = iris_ds['feature_names']
print("data.feature_names  = {0}".format(feature_names))

target = iris_ds['target']
print("data.target.shape = {0}".format(target.shape))

def plot_iris_projection(x_index, y_index):
    # plt.scatter one type of iris flower with one color.
    types_count = 3
    for t,marker,c in zip(range(types_count),'>ox', 'rgb'):
        plt.scatter(data[target==t,x_index],
                    data[target==t,y_index],
                    marker=marker,c=c)
    plt.xlabel(feature_names[x_index])
    plt.ylabel(feature_names[y_index])

pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
for i,(x_index,y_index) in enumerate(pairs):
    plt.subplot(2,3,i+1)
    plot_iris_projection(x_index, y_index)
plt.show()