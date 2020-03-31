# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/31 0031 9:39
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# 选择最佳的轮廓系数
# 知道每次聚类的轮廓系数
# 聚类后图像的分布
n_clusters = 4
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
fig, (ax1, ax2) = plt.subplots(1, 2)
# 画布尺寸
fig.set_size_inches(18, 7)

# 轮廓系数的取值在-1,1之间，但我们希望轮廓系数是大于0的
# 太长的横坐标不利于可视化
ax1.set_xlim([-0.1, 1])

# 设定纵坐标，通常来说，纵坐标是从0开始，最大值为X.shape[0]
# 但为了不同的簇能够有一定的间隙，便于观察
# (n_clusters + 1) * 10作为间隔
ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])

# 建模，调用聚类好的标签
clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
cluster_labels = clusterer.labels_

# 平均轮廓系数
silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)
# 每个样本的轮廓系数
sample_silhouette_values = silhouette_samples(X, cluster_labels)

# 设定y轴上的初始取值
y_lower = 10
# 对每一个簇进行循环
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    #     sort会直接改掉原数据的顺序
    ith_cluster_silhouette_values.sort()
    #    每一个簇中的样本
    size_cluster_i = ith_cluster_silhouette_values.shape[0]

    y_upper = y_lower + size_cluster_i

    #   颜色，使用小数来调色
    color = cm.nipy_spectral(float(i) / n_clusters)

    ax1.fill_betweenx(np.arange(y_lower, y_upper)
                      , ith_cluster_silhouette_values
                      , facecolor=color
                      , alpha=0.7
                      )
    # 显示簇的标签
    # 标签的横坐标，纵坐标，标签的内容
    ax1.text(-0.05
             , y_lower + 0.5 * size_cluster_i
             , str(i))
    # 更新y_lower
    y_lower = y_upper + 10

# 标题，横坐标名称，纵坐标名称
ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# 将平均轮廓系数一虚线的形式放入图中
# 画垂直线
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

# y轴不显示刻度
# ax1.set_yticks([])

ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 对子图2进行处理
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
# 画数据
ax2.scatter(X[:, 0], X[:, 1]
            , marker='o'
            , s=8
            , c=colors
            )

centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
# 画质心
ax2.scatter(centers[:, 0], centers[:, 1], marker='x',
            c="red", alpha=1, s=200)
# 为子图2设置标题
ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

# 为整个图设置标题
plt.suptitle(("Silhouette analysis for KMeans clustering on sample data"
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')
plt.show()
