# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/29 0029 9:07

from sklearn.linear_model import Ridge, LinearRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing, load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set = fetch_california_housing()
X = pd.DataFrame(data_set.data)
y = data_set.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)


def lasso_regression_v1():
    reg = Lasso()
    reg.fit(x_train, y_train)
    s = reg.score(x_test, y_test)
    print(s)

    w = reg.coef_  # w,系数向量
    b = reg.intercept_
    print("权重项：", w)
    print("偏置项：", b)

    y_pre = reg.predict(x_test)
    mse = mean_squared_error(y_pre, y_test)
    print("mse：", mse)

    r2_s = r2_score(y_pre, y_test)
    print("r2_score：", r2_s)


def lasso_regression_v2():
    alpharange = np.logspace(-10, -2, 2000, base=10)

    reg = LassoCV(alphas=alpharange, cv=5)
    reg.fit(x_train, y_train)

    best_alpha = reg.alpha_
    print("best_alpha: ", best_alpha)

    w = reg.coef_  # w,系数向量
    b = reg.intercept_
    print("权重项：", w)
    print("偏置项：", b)


def main():
    lasso_regression_v1()
    lasso_regression_v2()


if __name__ == '__main__':
    main()
