# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/31 0031 12:27

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
y = iris.target
X = iris.data
pd.DataFrame(X)

# 调用PCA
pca = PCA(n_components=2)  # 实例化
X_dr = pca.fit_transform(X)

colors = ['red', 'black', 'orange']

plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_dr[y == i, 0], X_dr[y == i, 1], alpha=0.7, c=colors[i], label=iris.target_names[i])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()
