# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/29 0029 9:47

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

rnd = np.random.RandomState(42)  # 设置随机数种子
X = rnd.uniform(-3, 3, size=100)
y = np.sin(X) + rnd.normal(size=len(X)) / 3

# 将X升维，准备好放入 sklearn中
X = X.reshape(-1, 1)

# 创建测试数据，均匀分布在训练集X的取值范围内的一千个点
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

d = 6
# 和上面展示一致的建模流程

ploy_reg = PolynomialFeatures(degree=d)
X_ = ploy_reg.fit_transform(X)

linear_reg = LinearRegression()
PolyR_ = linear_reg.fit(X_, y)

linear_reg2 = LinearRegression()
LinearR = linear_reg2.fit(X, y)

print("线性回归系数：", linear_reg2.coef_)
print("线性回归偏置：", linear_reg2.intercept_)

print("多项式回归系数：", linear_reg.coef_)
print("多项式回归偏置：", linear_reg.intercept_)

Poly_line = ploy_reg.fit_transform(line)

# 放置画布
fig2, ax2 = plt.subplots(1)

# 将测试数据带入predict接口，获得模型的拟合效果并进行绘制
ax2.plot(line, LinearR.predict(line), linewidth=2, color='green', label="linear regression")
ax2.plot(line, PolyR_.predict(Poly_line), linewidth=2, color='red', label="Polynomial regression")

# 将原数据上的拟合绘制在图像上
ax2.plot(X[:, 0], y, 'o', c='k')

# 其他图形选项
ax2.legend(loc="best")
ax2.set_ylabel("Regression output")
ax2.set_xlabel("Input feature")
ax2.set_title("Linear Regression ordinary vs poly")
plt.tight_layout()
plt.show()
