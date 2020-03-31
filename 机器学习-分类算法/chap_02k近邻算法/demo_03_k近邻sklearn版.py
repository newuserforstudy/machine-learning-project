# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/26 0026 18:17


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


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


def main():
    file_name = "C:\\Users\\Administrator\\Desktop\\" \
                "7日内日完成任务\\机器学习实战源码\\" \
                "Machine-Learning-in-Action-Python3-master\\" \
                "kNN_Project1\\datingTestSet.txt"
    data_feature, data_labels = load_data(file_name)

    s = StandardScaler()
    data_feature = s.fit_transform(data_feature)

    x_train, x_test, y_train, y_test = train_test_split(data_feature, data_labels, test_size=0.25,random_state=10)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)

    # 模型评价
    print("Test set score: {:.6f}".format(knn.score(x_test, y_test)))


if __name__ == '__main__':
    main()
