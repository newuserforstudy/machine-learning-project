# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/26 0026 19:57

from sklearn import tree
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

wine = load_wine()
print(wine.data.shape)
print(wine.target)

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)


def decision_tree_id3():
    print("--------------使用信息增益ID3----------------------")
    tree_model = tree.DecisionTreeClassifier(criterion="entropy", random_state=10)
    tree_model.fit(x_train, y_train)
    test_predict = tree_model.predict(x_test)
    acc_score = accuracy_score(y_test, test_predict)
    pre_score = precision_score(y_test, test_predict, average="weighted")

    recall_s = recall_score(y_test, test_predict, average='weighted')
    f1 = f1_score(y_true=y_test, y_pred=test_predict, average='weighted')
    cf = confusion_matrix(y_test, test_predict)

    print("精度：", acc_score)
    print("准确率：", pre_score)
    print("召回率：", recall_s)
    print("f1值：", f1)
    print("混淆矩阵：", cf)


def decision_tree_c4_5():
    print("--------------使用基尼系数C4.5----------------------")
    # tree_model = tree.DecisionTreeClassifier(criterion="gini")
    tree_model = tree.DecisionTreeClassifier(random_state=10)
    tree_model.fit(x_train, y_train)
    test_predict = tree_model.predict(x_test)
    acc_score = accuracy_score(y_test, test_predict)
    pre_score = precision_score(y_test, test_predict, average="weighted")

    recall_s = recall_score(y_test, test_predict, average='weighted')
    f1 = f1_score(y_true=y_test, y_pred=test_predict, average='weighted')
    cf = confusion_matrix(y_test, test_predict)

    print("精度：", acc_score)
    print("准确率：", pre_score)
    print("召回率：", recall_s)
    print("f1值：", f1)
    print("混淆矩阵：", cf)


def decision_tree_pre_cut():
    print("--------------决策树剪枝----------------------")
    tree_model = tree.DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=5, min_samples_leaf=10,
                                             random_state=10)
    tree_model.fit(x_train, y_train)
    test_predict = tree_model.predict(x_test)
    acc_score = accuracy_score(y_test, test_predict)
    pre_score = precision_score(y_test, test_predict, average="weighted")

    recall_s = recall_score(y_test, test_predict, average='weighted')
    f1 = f1_score(y_true=y_test, y_pred=test_predict, average='weighted')
    cf = confusion_matrix(y_test, test_predict)

    print("精度：", acc_score)
    print("准确率：", pre_score)
    print("召回率：", recall_s)
    print("f1值：", f1)
    print("混淆矩阵：", cf)


def main():
    decision_tree_id3()
    decision_tree_c4_5()
    decision_tree_pre_cut()


if __name__ == '__main__':
    main()
