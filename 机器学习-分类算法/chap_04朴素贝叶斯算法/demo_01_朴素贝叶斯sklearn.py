# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/28 0028 8:04
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

digits = load_digits()  # 手写数据集
x, y = digits.data, digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=420)


def naive_bayes_gaussian():
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    acc_score = gnb.score(x_test, y_test)  # 返回预测的精确性 accuracy

    print(acc_score)
    # 查看预测结果
    Y_pred = gnb.predict(x_test)

    # 查看预测的概率结果
    prob = gnb.predict_proba(x_test)


def naive_bayes_bernoulli():
    gnb = BernoulliNB()
    gnb.fit(x_train, y_train)
    acc_score = gnb.score(x_test, y_test)  # 返回预测的精确性 accuracy
    print(acc_score)


def naive_bayes_multinomial():
    gnb = MultinomialNB()
    gnb.fit(x_train, y_train)
    acc_score = gnb.score(x_test, y_test)  # 返回预测的精确性 accuracy
    print(acc_score)


def main():
    naive_bayes_gaussian()
    naive_bayes_bernoulli()
    naive_bayes_multinomial()


if __name__ == '__main__':
    main()
