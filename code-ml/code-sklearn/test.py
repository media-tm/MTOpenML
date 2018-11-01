#
# Python 3.6
#
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()

print(iris.keys())
print("iris.keys()  = {0}".format(iris.keys()))

raw_data = iris['data']
print("iris.data.shape  = {0}".format(raw_data.shape))

feature_names = iris['feature_names']
print("data.feature_names  = {0}".format(feature_names))

raw_target = iris['target']
print("data.target.shape = {0}".format(raw_target.shape))

def plot_iris_projection(x_index, y_index):
    # plt.scatter one type of iris flower with one color.
    types_count = 3
    for t,marker,c in zip(range(types_count),'>ox', 'rgb'):
        plt.scatter(dst_data[dst_target==t,x_index],
                    dst_data[dst_target==t,y_index],
                    marker=marker,c=c)
    if(dst_data.shape[1] == 4):
        plt.xlabel(feature_names[x_index])
        plt.ylabel(feature_names[y_index])

plt.figure(figsize=(16, 9))
dst_data   = raw_data
dst_target = raw_target
plt.subplot(2, 2, 1)
plot_iris_projection(x_index=0, y_index=1)
plt.subplot(2, 2, 2)
plot_iris_projection(x_index=0, y_index=2)

# supervised dimensionality reduction
dst_data = LinearDiscriminantAnalysis(n_components=2).fit_transform(raw_data, raw_target)
ax = plt.subplot(2, 2, 3)
ax.set_title('LDA(n_components=2)')
plot_iris_projection(x_index=0, y_index=1)
print(dst_data.shape)

# supervised dimensionality reduction
dst_data = LinearDiscriminantAnalysis(n_components=3).fit_transform(raw_data, raw_target)
ax = plt.subplot(2, 2, 4)
ax.set_title('LDA(n_components=3)')
plot_iris_projection(x_index=0, y_index=1)
print(dst_data.shape)

#Adjust subgraph spacing
plt.subplots_adjust(wspace =0.3, hspace =0.4) 

plt.show()