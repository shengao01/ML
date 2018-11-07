from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# data = [[2, 8, 4, 5],
#         [6, 3, 0, 8],
#         [5, 4, 9, 1]]
#
# pca = PCA(n_components=2)
# result = pca.fit_transform(data)
# print(result)

# 对鸢尾花数据进行降维
iris = load_iris()
pca = PCA(n_components=2)
print(iris.data)
result = pca.fit_transform(iris.data)
print(result)