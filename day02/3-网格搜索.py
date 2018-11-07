import numpy as np
import pandas as pd

# 1、导入numpy和pandas
# 2、加载数据文件（read_csv）
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("C:/Users/Leon/Desktop/AI/data/FBlocation/train.csv")
# 3、缩小数据范围（data.query）
data = data.query("x > 3 & x < 3.25 & y > 3 & y < 3.25")
# 4、剔除掉入住率比较低的样本
# 找到入住次数比较少的place_id
place_count = data.groupby("place_id").aggregate(np.count_nonzero)
# 剔除掉入住次数小于3
rf = place_count[place_count["row_id"] > 3].reset_index()
# 从原始数据里面提取出rf中place_id对应的数据
data = data[data["place_id"].isin(rf["place_id"])]
# print(data)
# 5、分割数据集（train_test_split）
# 特征值
x = data.drop(["row_id", "time", "place_id"], axis=1)
# 目标值
y = data["place_id"]
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 6、对数据集进行标准化
ss = StandardScaler()
# 对训练集的标准化
x_train = ss.fit_transform(x_train)
# 对测试集标准化, 以训练集的均值做转换
x_test = ss.transform(x_test)
# 7、KNeighborsClassifier训练模型
knn = KNeighborsClassifier(n_neighbors=5)

param_grid = {"n_neighbors": [1, 3, 5]}
gscv = GridSearchCV(estimator=knn, param_grid=param_grid, cv=2)
gscv.fit(x_train, y_train)
print("最好的结果", gscv.best_score_)
print("最好的参数模型", gscv.best_estimator_)
print("交叉验证的结果", gscv.cv_results_)