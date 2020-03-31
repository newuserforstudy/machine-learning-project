# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/31 0031 9:14
from xgboost import XGBClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

reg = XGBClassifier(n_estimators=100).fit(x_train, y_train)  # 训练
s = reg.score(x_test, y_test)
print(s)
