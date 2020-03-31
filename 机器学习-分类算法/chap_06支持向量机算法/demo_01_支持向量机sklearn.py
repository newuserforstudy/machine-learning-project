# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/28 0028 8:35
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="rainbow")
plt.xticks([])
plt.yticks([])
plt.show()


def plot_svc_decision_function(model, ax=None):
    if ax is None:
        ax = plt.gca()
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    x_local = np.linspace(x_lim[0], x_lim[1], 30)
    y_local = np.linspace(y_lim[0], y_lim[1], 30)
    y_local, x_local = np.meshgrid(y_local, x_local)
    xy = np.vstack([x_local.ravel(), y_local.ravel()]).T

    dist = model.decision_function(xy).reshape(x_local.shape)
    ax.contour(x_local, y_local, dist, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.show()


# 则整个绘图过程可以写作：
svm = SVC(kernel="linear").fit(x, y)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="rainbow")
plot_svc_decision_function(svm)
