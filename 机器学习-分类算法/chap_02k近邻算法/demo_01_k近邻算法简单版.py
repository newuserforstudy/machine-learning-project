# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/26 0026 16:53
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_name):
    data_feature = []
    data_labels = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            data_feature.append(line[0:3])
            data_labels.append(line[-1])

    for i in range(len(data_labels)):
        if data_labels[i] == "largeDoses":
            data_labels[i] = 0
        elif data_labels[i] == "smallDoses":
            data_labels[i] = 1
        elif data_labels[i] == "didntLike":
            data_labels[i] = 2

    data_feature = np.asarray(data_feature).astype(np.float32)
    data_labels = np.asarray(data_labels).astype(np.int32)

    return data_feature, data_labels


def data_min_max(data_feature):
    print(data_feature.shape)
    data_set = np.zeros_like(data_feature)
    col = data_feature.shape[1]
    row = data_feature.shape[0]
    for i in range(col):
        # print(i)
        min_value = np.min(data_feature[:, i])
        max_value = np.max(data_feature[:, i])

        for j in range(row):
            data_set[j, i] = (data_feature[j, i] - min_value) / (max_value - min_value)

    print(data_set[0:5])
    return data_set


def split_train_test(data_feature, data_labels, split_radio=0.75):

    # data_feature = data_min_max(data_feature)
    indices = np.random.permutation(len(data_feature))

    data_feature = data_feature[indices]
    data_labels = data_labels[indices]

    threshold = np.int32(len(data_feature) * split_radio)

    train_x, train_y = data_feature[0:threshold, :], data_labels[0:threshold]
    test_x, test_y = data_feature[threshold:len(data_feature), :], data_labels[threshold:len(data_feature)]
    print("训练集样本数：", len(train_x))
    print("测试集样本数：", len(test_x))

    return train_x, train_y, test_x, test_y


def calculate_dist(vec_a, vec_b):
    return np.sum(np.square(vec_a - vec_b))


def knn_model(train_x, train_y, test_x, test_y, top_k=20):
    correct_num = 0
    for i in range(len(test_x)):
        dist_result = []
        for train in train_x:
            dist_result.append(calculate_dist(test_x[i], train))

        sorted_result = sorted(dist_result, reverse=True)[0:top_k]
        print(sorted_result)
        sorted_labels = []

        for dist in sorted_result:
            sorted_labels.append(train_y[dist_result.index(dist)])

        each_label_num_dict = Counter(sorted_labels)
        print(each_label_num_dict)
        predict = sorted(each_label_num_dict.items(), key=lambda item: item[1], reverse=True)[0][0]

        # print("预测结果为：", predict)
        test_label = test_y[i]
        # print("实际结果为：", test_label)

        if predict == test_label:
            correct_num += 1
        # print(correct_num)
    accuracy = correct_num / len(test_x)

    print(accuracy)


def main():
    file_name = "C:\\Users\\Administrator\\Desktop\\" \
                "7日内日完成任务\\机器学习实战源码\\" \
                "Machine-Learning-in-Action-Python3-master\\" \
                "kNN_Project1\\datingTestSet.txt"
    data_feature, data_labels = load_data(file_name)
    s = StandardScaler()
    data_feature = s.fit_transform(data_feature)
    x_train, x_test, y_train, y_test = split_train_test(data_feature, data_labels)

    knn_model(x_train, x_test, y_train, y_test, top_k=10)


if __name__ == '__main__':
    main()
