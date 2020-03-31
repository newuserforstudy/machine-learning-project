# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/28 0028 7:17

import cv2
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 二值化
def image2binary(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


def binary_features(train_set):
    features_local = []

    for img in train_set:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        img_b = image2binary(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features_local.append(img_b)

    features_local = np.array(features_local)
    features_local = np.reshape(features_local, (-1, feature_len))

    return features_local


class Tree(object):
    def __init__(self, node_type, class_leaf=None, feature=None):
        self.node_type = node_type  # 节点类型（internal或leaf）
        self.dict = {}  # dict的键表示特征Ag的可能值ai，值表示根据ai得到的子树
        self.class_leaf = class_leaf  # 叶节点表示的类，若是内部节点则为none
        self.feature = feature  # 表示当前的树即将由第feature个特征划分（即第feature特征是使得当前树中信息增益最大的特征）

    def add_tree(self, key, tree_local):
        self.dict[key] = tree_local

    def predict(self, features_local):
        if self.node_type == 'leaf' or (features_local[self.feature] not in self.dict):
            return self.class_leaf

        tree_local = self.dict.get(features_local[self.feature])
        return tree_local.predict(features_local)


# 计算数据集x的经验熵H(x)
def calc_ent(x):
    x_value_list = set([x[i_local] for i_local in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        log_p = np.log2(p)
        ent -= p * log_p

    return ent


# 计算条件熵H(y/x)
def calc_condition_ent(x, y):
    x_value_list = set([x[i_local] for i_local in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent


# 计算信息增益
def calculate_entropy_gain(x, y):
    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


# ID3算法
def recurse_train(train_set, train_label, features_local):
    leaf = 'leaf'
    internal = 'internal'

    # 步骤1——如果训练集train_set中的所有实例都属于同一类Ck
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(leaf, class_leaf=label_set.pop())

    # 步骤2——如果特征集features为空
    # 计算每一个类出现的个数
    class_len = [(i_local, len(list(filter(lambda x: x == i_local, train_label)))) for i_local in range(class_num)]
    (max_class, max_len) = max(class_len, key=lambda x: x[1])

    if len(features_local) == 0:
        return Tree(leaf, class_leaf=max_class)

    # 步骤3——计算信息增益,并选择信息增益最大的特征
    max_feature = 0
    max_gda = 0
    d = train_label
    for feature in features_local:
        # print(type(train_set))
        a = np.array(train_set[:, feature].flat)  # 选择训练集中的第feature列（即第feature个特征）
        gda = calculate_entropy_gain(a, d)
        if gda > max_gda:
            max_gda, max_feature = gda, feature

    # 步骤4——信息增益小于阈值
    if max_gda < epsilon:
        return Tree(leaf, class_leaf=max_class)

    # 步骤5——构建非空子集
    sub_features = list(filter(lambda x: x != max_feature, features_local))
    tree_local = Tree(internal, feature=max_feature)

    max_feature_col = np.array(train_set[:, max_feature].flat)
    feature_value_list = set(
        [max_feature_col[i_local] for i_local in range(max_feature_col.shape[0])])  # 保存信息增益最大的特征可能的取值 (shape[0]表示计算行数)
    for feature_value in feature_value_list:

        index = []
        for i_local in range(len(train_label)):
            if train_set[i_local][max_feature] == feature_value:
                index.append(i_local)

        sub_train_set = train_set[index]
        sub_train_label = train_label[index]

        sub_tree = recurse_train(sub_train_set, sub_train_label, sub_features)
        tree_local.add_tree(feature_value, sub_tree)

    return tree_local


def train(train_set, train_label, features_local):
    return recurse_train(train_set, train_label, features_local)


def predict(test_set, tree_local):
    result = []
    for features_local in test_set:
        tmp_predict = tree_local.predict(features_local)
        result.append(tmp_predict)
    return np.array(result)


class_num = 10  # MINST数据集有10种labels，分别是“0,1,2,3,4,5,6,7,8,9”
feature_len = 784  # MINST数据集每个image有28*28=784个特征（pixels）
epsilon = 0.001  # 设定阈值

if __name__ == '__main__':

    print("Start read data...")

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv', header=0)  # 读取csv数据
    data = raw_data.values

    imgs = data[::, 1::]
    features = binary_features(imgs)  # 图片二值化(很重要，不然预测准确率很低)
    labels = data[::, 0]

    # 避免过拟合，采用交叉验证，随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=0)
    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    # 通过ID3算法生成决策树
    print('Start training...')
    tree = train(train_features, train_labels, list(range(feature_len)))
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = predict(test_features, tree)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    # print("预测的结果为：")
    # print(test_predict)
    for i in range(len(test_predict)):
        if test_predict[i] is None:
            test_predict[i] = epsilon
    score = accuracy_score(test_labels, test_predict)
    print("The accuracy score is %f" % score)
