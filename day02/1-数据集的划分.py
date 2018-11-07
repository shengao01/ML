import numpy as np

# 特征值
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# x = np.arange(10).reshape([5,2])
# print(x)
# # 目标值
# y = range(5)
# print(y)

iris = load_iris()
# 特征值
x = iris.data
# 目标值
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# 训练集的特征值
# print(x_train)
# 训练集的目标值
# print(y_train)
# 测试集的特征值
print(x_test)
print(y_test)
