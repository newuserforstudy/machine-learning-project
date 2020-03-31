# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/28 0028 8:52

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

data = load_digits()
X, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

"""
["linear", "poly", "rbf", "sigmoid", "precomputed"]
"""


def svm_v1():
    svm = SVC(kernel="linear", max_iter=50)
    svm.fit(x_train, y_train)
    print(svm.score(x_test, y_test))


def svm_v2():
    svm = SVC(C=0.5, kernel="linear", max_iter=50)
    svm.fit(x_train, y_train)
    print(svm.score(x_test, y_test))


def svm_v3():
    svm = SVC(C=0.5, kernel="rbf", max_iter=50)
    svm.fit(x_train, y_train)
    print(svm.score(x_test, y_test))


def main():
    svm_v1()
    svm_v2()
    svm_v3()


if __name__ == '__main__':
    main()
