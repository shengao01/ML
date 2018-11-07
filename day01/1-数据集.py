from sklearn.datasets import load_iris, load_boston

iris = load_iris()

# 数据集的描述
# print(iris.DESCR)
# 特征名
# print(iris.feature_names)
# 特征值
# print(iris.data)

# 目标名
# print(iris.target_names)
# 目标值(标签值)
# print(iris.target)

# 加载波士顿房价数据集
boston = load_boston()
# print(boston.DESCR)
# print(boston.feature_names)
# print(boston.data)

# print(boston.target_names) 没有target_names
# 目标值
print(boston.target)