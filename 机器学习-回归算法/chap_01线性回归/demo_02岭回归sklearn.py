# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/29 0029 8:41

from sklearn.linear_model import Ridge, LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing, load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set = fetch_california_housing()
X = pd.DataFrame(data_set.data)
y = data_set.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)


def ridge_regression_v1():
    # reg = Ridge()
    # reg.fit(x_train, y_train)

    alpharange = np.arange(1, 1001, 100)
    ridge, lr = [], []
    for alpha in alpharange:
        reg = Ridge(alpha=alpha)
        linear = LinearRegression()
        regs = cross_val_score(reg, X, y, cv=5, scoring="r2").mean()
        linears = cross_val_score(linear, X, y, cv=5, scoring="r2").mean()
        ridge.append(regs)
        lr.append(linears)
    plt.plot(alpharange, ridge, color="red", label="Ridge")
    plt.plot(alpharange, lr, color="orange", label="LR")
    plt.title("Mean")
    plt.legend()
    plt.show()

    # 模型方差如何变化？
    alpharange = np.arange(1, 1001, 100)
    ridge, lr = [], []
    for alpha in alpharange:
        reg = Ridge(alpha=alpha)
        linear = LinearRegression()
        varR = cross_val_score(reg, X, y, cv=5, scoring="r2").var()
        varLR = cross_val_score(linear, X, y, cv=5, scoring="r2").var()
        ridge.append(varR)
        lr.append(varLR)
    plt.plot(alpharange, ridge, color="red", label="Ridge")
    plt.plot(alpharange, lr, color="orange", label="LR")
    plt.title("Variance")
    plt.legend()
    plt.show()

    # 使用岭回归来进行建模
    reg = Ridge(alpha=100).fit(x_train, y_train)
    s = reg.score(x_test, y_test)
    print(s)

    # 细化一下学习曲线
    alpharange = np.arange(1, 201, 10)
    ridge, lr = [], []
    for alpha in alpharange:
        reg = Ridge(alpha=alpha)
        linear = LinearRegression()
        regs = cross_val_score(reg, X, y, cv=5, scoring="r2").mean()
        linears = cross_val_score(linear, X, y, cv=5, scoring="r2").mean()
        ridge.append(regs)
        lr.append(linears)
    plt.plot(alpharange, ridge, color="red", label="Ridge")
    plt.plot(alpharange, lr, color="orange", label="LR")
    plt.title("Mean")
    plt.legend()
    plt.show()


data_set_boston = load_boston()
x_boston = data_set_boston.data
y_boston = data_set_boston.target
x_boston_train, x_boston_test, y_boston_train, y_boston_test = train_test_split(x_boston, y_boston, test_size=0.3,
                                                                                random_state=420)


def ridge_regression_v2():
    # 先查看方差的变化
    alpharange = np.arange(1, 1001, 100)
    ridge, lr = [], []
    for alpha in alpharange:
        reg = Ridge(alpha=alpha)
        linear = LinearRegression()
        varR = cross_val_score(reg, x_boston, y_boston, cv=5, scoring="r2").var()
        varLR = cross_val_score(linear, x_boston, y_boston, cv=5, scoring="r2").var()
        ridge.append(varR)
        lr.append(varLR)
    plt.plot(alpharange, ridge, color="red", label="Ridge")
    plt.plot(alpharange, lr, color="orange", label="LR")
    plt.title("Variance")
    plt.legend()
    plt.show()

    # 查看R2的变化
    alpharange = np.arange(1, 1001, 100)
    ridge, lr = [], []
    for alpha in alpharange:
        reg = Ridge(alpha=alpha)
        linear = LinearRegression()
        regs = cross_val_score(reg, x_boston, y_boston, cv=5, scoring="r2").mean()
        linears = cross_val_score(linear, x_boston, y_boston, cv=5, scoring="r2").mean()
        ridge.append(regs)
        lr.append(linears)
    plt.plot(alpharange, ridge, color="red", label="Ridge")
    plt.plot(alpharange, lr, color="orange", label="LR")
    plt.title("Mean")
    plt.legend()
    plt.show()

    # 细化学习曲线
    alpharange = np.arange(100, 300, 10)
    ridge, lr = [], []
    for alpha in alpharange:
        reg = Ridge(alpha=alpha)
        linear = LinearRegression()
        regs = cross_val_score(reg, X, y, cv=5, scoring="r2").mean()
        linears = cross_val_score(linear, X, y, cv=5, scoring="r2").mean()
        ridge.append(regs)
        lr.append(linears)
    plt.plot(alpharange, ridge, color="red", label="Ridge")
    plt.plot(alpharange, lr, color="orange", label="LR")
    plt.title("Mean")
    plt.legend()
    plt.show()


def ridge_regression_cv():
    Ridge_ = RidgeCV(alphas=np.arange(1, 1001, 100)
                     # ,scoring="neg_mean_squared_error"
                     , store_cv_values=True
                     # ,cv=5
                     ).fit(X, y)
    # 调用所有交叉验证的结果
    cv_values = Ridge_.cv_values_
    cv_a = Ridge_.alpha_
    print(cv_a)


def main():
    ridge_regression_v1()
    ridge_regression_v2()
    ridge_regression_cv()


if __name__ == '__main__':
    main()
