# Official Reference Document
# http://scikit-learn.org/stable/modules/preprocessing.html

from numpy import vstack, array, nan
from numpy import log1p

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer

from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sys, getopt

opts, args = getopt.getopt(sys.argv[1:], "hm:")
method_reg = "feature-select"
for op, value in opts:
    if op == "-m":
        if value == "pre-proc-1":
            method_reg = value
        elif value == "pre-proc-2":
            method_reg = value
        elif value == "dimen-reduction":
            method_reg = value
    elif op == "-h":
        print("-m pre-proc-1/pre-proc-2/feature-select/dimen-reduction")
        sys.exit()

#
# Python 3.6
#

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
plt.subplot(2, 4, 1)
plot_iris_projection(x_index=0, y_index=1)
plt.subplot(2, 4, 2)
plot_iris_projection(x_index=0, y_index=2)
plt.subplot(2, 4, 3)
plot_iris_projection(x_index=0, y_index=3)
plt.subplot(2, 4, 4)
plot_iris_projection(x_index=1, y_index=2)
if method_reg == "pre-proc-1":
    dst_data = StandardScaler().fit_transform(raw_data)
    ax = plt.subplot(2, 4, 1+4)
    ax.set_title('StandardScaler()')
    plot_iris_projection(x_index=0, y_index=1)

    dst_data = MinMaxScaler().fit_transform(raw_data)
    ax = plt.subplot(2, 4, 2+4)
    ax.set_title('MinMaxScaler()')
    plot_iris_projection(x_index=0, y_index=2)

    dst_data = Normalizer().fit_transform(raw_data)
    ax = plt.subplot(2, 4, 3+4)
    ax.set_title('Normalizer()')
    plot_iris_projection(x_index=0, y_index=3)

    dst_data = Binarizer(threshold=3).fit_transform(raw_data)
    ax = plt.subplot(2, 4, 4+4)
    ax.set_title('Binarizer(threshold=3)')
    plot_iris_projection(x_index=1, y_index=2)
elif method_reg == "pre-proc-2":
    # Subcode the target value of the IRIS data set
    dst_target = OneHotEncoder().fit_transform(raw_target.reshape((-1,1)))
    ax = plt.subplot(2, 4, 1+4)
    ax.set_title('OneHotEncoder()')
    print(dst_target.shape)
    dst_target = raw_target     # restore raw_target
    plot_iris_projection(x_index=0, y_index=1)

    dst_data = Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), raw_data[:149])))
    ax = plt.subplot(2, 4, 2+4)
    ax.set_title('Imputer()')
    plot_iris_projection(x_index=0, y_index=2)

    dst_data = PolynomialFeatures().fit_transform(raw_data)
    ax = plt.subplot(2, 4, 3+4)
    ax.set_title('PolynomialFeatures()')
    plot_iris_projection(x_index=0, y_index=3)

    dst_data = FunctionTransformer(log1p).fit_transform(raw_data)
    ax = plt.subplot(2, 4, 4+4)
    ax.set_title('FunctionTransformer()')
    plot_iris_projection(x_index=1, y_index=2)
elif method_reg == "feature-select":
    dst_data = StandardScaler().fit_transform(raw_data)
    # variance selection method
    # parameter threshold is the threshold of variance
    dst_data = VarianceThreshold(threshold=3).fit_transform(raw_data)
    ax = plt.subplot(2, 4, 1+4)
    ax.set_title('VarianceThreshold()')
    plot_iris_projection(x_index=0, y_index=0)
    print(dst_data.shape)

    # Chi-square test
    dst_data = SelectKBest(chi2, k=2).fit_transform(raw_data, raw_target)
    # Correlation coefficient method
    # dst_data = SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(raw_data, raw_target)
    ax = plt.subplot(2, 4, 2+4)
    ax.set_title('SelectKBest()')
    plot_iris_projection(x_index=0, y_index=1)
    print(dst_data.shape)

    # recursive feature elimination
    dst_data = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(raw_data, raw_target)
    ax = plt.subplot(2, 4, 3+4)
    ax.set_title('RFE()')
    plot_iris_projection(x_index=0, y_index=1)
    print(dst_data.shape)

    # Penalty-based feature selection
    # dst_data = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(raw_data, raw_target)
    # Feature selection based on tree model
    dst_data = SelectFromModel(GradientBoostingClassifier()).fit_transform(raw_data, raw_target)
    ax = plt.subplot(2, 4, 4+4)
    ax.set_title('SelectFromModel()')
    plot_iris_projection(x_index=0, y_index=1)
    print(dst_data.shape)

elif method_reg == "dimen-reduction":
    # unsupervised dimensionality reduction
    dst_data = PCA(n_components=2).fit_transform(raw_data)
    ax = plt.subplot(2, 4, 1+4)
    ax.set_title('PCA(n_components=2)')
    plot_iris_projection(x_index=0, y_index=1)
    print(dst_data.shape)

    # unsupervised dimensionality reduction
    dst_data = PCA(n_components=3).fit_transform(raw_data)
    ax = plt.subplot(2, 4, 2+4)
    ax.set_title('PCA(n_components=3)')
    plot_iris_projection(x_index=0, y_index=1)
    print(dst_data.shape)

    # supervised dimensionality reduction
    dst_data = LinearDiscriminantAnalysis(n_components=2).fit_transform(raw_data, raw_target)
    ax = plt.subplot(2, 4, 3+4)
    ax.set_title('LDA(n_components=2)')
    plot_iris_projection(x_index=0, y_index=1)
    print(dst_data.shape)

    # supervised dimensionality reduction
    dst_data = LinearDiscriminantAnalysis(n_components=3).fit_transform(raw_data, raw_target)
    ax = plt.subplot(2, 4, 4+4)
    ax.set_title('LDA(n_components=3)')
    plot_iris_projection(x_index=0, y_index=1)
    print(dst_data.shape)

#Adjust subgraph spacing
plt.subplots_adjust(wspace =0.3, hspace =0.4) 

plt.show()



