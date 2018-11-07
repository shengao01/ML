from matplotlib import pyplot as plt

# 1、获取数据
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

iris = load_iris()
# 2、k-means聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(iris.data)
predict = kmeans.predict(iris.data)
print(predict)

# 3、聚类结果显示
plt.figure(figsize=[4, 4])

colors = ["red", "green", "blue"]
c = [colors[k] for k in predict]
# 选两个特征作为x， y
plt.scatter(iris.data[:,0], iris.data[:,2], c=c)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()

score = silhouette_score(X=iris.data, labels=predict)
print(score)