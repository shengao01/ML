import pandas as pd
import numpy as np

# 1、网上获取数据（工具pandas）
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

column_names = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
                'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
            names=column_names)
# 2、数据缺失值处理
data.replace("?", np.nan, inplace=True)
data.dropna(inplace=True)
print(data)
# 3、数据分割
# 特征值
x = data[column_names[1:10]]
# 目标值
y = data[column_names[10]]
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 4、标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 5、LogisticRegression估计器流程
lr = LogisticRegression()
lr.fit(x_train, y_train)
predict = lr.predict(x_test)
print(predict)
score = lr.score(x_test, y_test)
print(score)


# 6、获取精确率和召回率
report = classification_report(y_true=y_test, y_pred=predict)
print(report)
