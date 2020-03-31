# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/31 0031 9:23
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabaz_score

n_clusters = 4
# 自己创建一个数据集
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

fig, ax1 = plt.subplots(1)
# fig是画布，ax1是对象
ax1.scatter(X[:, 0], X[:, 1], marker='o', s=8)
plt.show()

color = ["red", "pink", "orange", "gray"]
fig1, ax1 = plt.subplots(1)

for i in range(4):
    ax1.scatter(X[y == i, 0], X[y == i, 1], marker='o', s=12, c=color[i])
plt.show()

# 聚类
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

# 聚类的结果
y_pred3 = cluster.labels_

# 质心
centroid = cluster.cluster_centers_

# 总距离平方和
inertia = cluster.inertia_

fig, ax1 = plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(X[y_pred3 == i, 0], X[y_pred3 == i, 1], marker='o', s=8, c=color[i])

ax1.scatter(centroid[:, 0], centroid[:, 1], marker="*", s=1000, c="black")
plt.show()

s = silhouette_score(X, y_pred3)
m = silhouette_samples(X, y_pred3).mean()
cs = calinski_harabaz_score(X, y_pred3)
