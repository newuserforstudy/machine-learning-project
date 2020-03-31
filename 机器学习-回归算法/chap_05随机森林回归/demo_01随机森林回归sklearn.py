# *_* coding:utf-8 *_*
# @author:sdh
# @Time : 2020/3/29 0029 11:52
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
boston = load_boston()
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
s = cross_val_score(regressor, boston.data, boston.target, cv=10, scoring="neg_mean_squared_error")
print(s)