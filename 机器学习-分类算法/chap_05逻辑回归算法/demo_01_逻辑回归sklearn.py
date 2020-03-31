# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/28 0028 8:19
from sklearn.linear_model import LogisticRegression as lr
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target


def logistic_regression_v1():
    lrl1 = lr(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
    lrl2 = lr(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)
    # 逻辑回归的重要属性coef_，查看每个特征所对应的参数
    lrl1 = lrl1.fit(X, y)
    print(lrl1.coef_, lrl1.intercept_)

    lrl2 = lrl2.fit(X, y)
    print(lrl2.coef_, lrl2.intercept_)


def logistic_regression_v2():
    l1 = []
    l2 = []
    l1test = []
    l2test = []
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)
    for i in np.linspace(0.05, 1, 19):
        lrl1 = lr(penalty="l1", solver="liblinear", C=i, max_iter=1000)
        lrl2 = lr(penalty="l2", solver="liblinear", C=i, max_iter=1000)

        lrl1.fit(x_train, y_train)
        l1.append(accuracy_score(lrl1.predict(x_train), y_train))
        l1test.append(accuracy_score(lrl1.predict(x_test), y_test))

        lrl2.fit(x_train, y_train)
        l2.append(accuracy_score(lrl2.predict(x_train), y_train))
        l2test.append(accuracy_score(lrl2.predict(x_test), y_test))

    # %%

    graph = [l1, l2, l1test, l2test]
    color = ["red", "black", "red", "black"]
    label = ["train_L1", "train_L2", "test_L1", "test_L2"]
    plt.figure(figsize=(6, 6))
    for i in range(len(graph)):
        plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
    plt.legend(loc=1)  # 图例的位置在哪里?1右上,2左上,3左下,4右下角
    plt.show()


def main():
    logistic_regression_v1()
    logistic_regression_v2()


if __name__ == '__main__':
    main()
