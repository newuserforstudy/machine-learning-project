# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/29 0029 11:13

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_val_score as CVS
import numpy as np

rnd = np.random.RandomState(42)  # 设置随机数种子
X = rnd.uniform(-3, 3, size=100)  # random.uniform，从输入的任意两个整数中取出size个随机数

# 生成y的思路：先使用NumPy中的函数生成一个sin函数图像，然后再人为添加噪音
y = np.sin(X) + rnd.normal(size=len(X)) / 3  # random.normal，生成size个服从正态分布的随机数

# 使用散点图观察建立的数据集是什么样子
plt.scatter(X, y, marker='o', c='k', s=20)
plt.show()

# 使用原始数据进行建模
X = X.reshape(-1, 1)
LinearR = LinearRegression().fit(X, y)
TreeR = DecisionTreeRegressor(random_state=0).fit(X, y)

# 创建测试数据：一系列分布在横坐标上的点
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

pred, score, var = [], [], []
binsrange = [2, 5, 10, 15, 20, 30]
for i in binsrange:
    # 实例化分箱类
    enc = KBinsDiscretizer(n_bins=i, encode="onehot")
    # 转换数据
    X_binned = enc.fit_transform(X)
    line_binned = enc.transform(line)
    # 建立模型
    LinearR_ = LinearRegression()
    # 全数据集上的交叉验证
    cvresult = CVS(LinearR_, X_binned, y, cv=5)
    score.append(cvresult.mean())
    var.append(cvresult.var())
    # 测试数据集上的打分结果
    pred.append(LinearR_.fit(X_binned, y).score(line_binned, np.sin(line)))

# 绘制图像
plt.figure(figsize=(6, 5))
plt.plot(binsrange, pred, c="orange", label="test")
plt.plot(binsrange, score, c="k", label="full data")
plt.plot(binsrange, score + np.array(var) * 0.5, c="red", linestyle="--", label="var")
plt.plot(binsrange, score - np.array(var) * 0.5, c="red", linestyle="--")
plt.legend()
plt.show()
