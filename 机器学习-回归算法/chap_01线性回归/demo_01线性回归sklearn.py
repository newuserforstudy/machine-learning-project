# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/29 0029 8:04
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

data_set = fetch_california_housing()
X = pd.DataFrame(data_set.data)
y = data_set.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)
print(sorted(sklearn.metrics.SCORERS.keys()))


def linear_regression_v1():
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_pre = reg.predict(x_test)
    w = reg.coef_  # w,系数向量
    b = reg.intercept_
    print("权重项：", w)
    print("偏置项：", b)

    mse = mean_squared_error(y_pre, y_test)
    print("mse：", mse)
    n_mse = cross_val_score(reg, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print(" n_mse：",  n_mse)

    plt.plot(range(len(y_test)), sorted(y_test), c="black", label="Data")
    plt.plot(range(len(y_pre)), sorted(y_pre), c="red", label="Predict")
    plt.legend()
    plt.show()


def linear_regression_v2():
    reg = LinearRegression(normalize=True)
    reg.fit(x_train, y_train)
    y_pre = reg.predict(x_test)
    w = reg.coef_  # w,系数向量
    b = reg.intercept_
    print("权重项：", w)
    print("偏置项：", b)

    mse = mean_squared_error(y_pre, y_test)
    print("mse：", mse)

    cv_mse = cross_val_score(reg, X, y, cv=10, scoring="neg_mean_squared_error").mean()
    print("cv_mse：", cv_mse)

    r2_score = cross_val_score(reg, X, y, cv=10, scoring="r2").mean()
    print("r2_score：", r2_score)
    plt.plot(range(len(y_test)), sorted(y_test), c="black", label="Data")
    plt.plot(range(len(y_pre)), sorted(y_pre), c="red", label="Predict")
    plt.legend()
    plt.show()


def main():
    linear_regression_v1()
    linear_regression_v2()


if __name__ == '__main__':
    main()
