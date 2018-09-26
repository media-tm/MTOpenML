#
# Python 3.6
#

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

iris = load_iris()

print(iris.keys())
print("iris.keys()  = {0}".format(iris.keys()))

data = iris['data']
print("iris.data.shape  = {0}".format(data.shape))

feature_names = iris['feature_names']
print("data.feature_names  = {0}".format(feature_names))

target = iris['target']
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

#Adjust subgraph spacing
plt.subplots_adjust(wspace =0.3, hspace =0.4)
plt.show()