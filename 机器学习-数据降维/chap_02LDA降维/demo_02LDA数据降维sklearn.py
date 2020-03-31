# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/31 0031 12:38

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets.samples_generator import make_classification

X, y = make_classification(n_samples=1000,
                           n_features=3,
                           n_redundant=0,
                           n_classes=3,
                           n_informative=2,
                           n_clusters_per_class=1,
                           class_sep=0.5,
                           random_state=10)
# fig = plt.figure()
#
# ax = Axes3D(fig, rect=(0, 0, 1, 1), elev=30, azim=20)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)


pca = PCA(n_components=2)
X_new = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()


lda = LinearDiscriminantAnalysis(n_components=2)
x_lda = lda.fit_transform(X,y)
plt.scatter(x_lda[:, 0], x_lda[:, 1], marker='o', c=y)
plt.show()
