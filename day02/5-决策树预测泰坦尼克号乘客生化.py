import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

# 1、pd读取数据
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
# 2、选择有影响的特征
# 区分特征值和目标值
x = data[["pclass", "age", "sex"]]
y = data[["survived"]]
# 3、填补缺失值age
x["age"].fillna(x["age"].mean(), inplace=True)
# 4、数据分割
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 5、转换成字典，对数据集特征抽取将有类别的特征（票类，性别）变成One-Hot编码形式
x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
dv = DictVectorizer()
# 特征名
x_train = dv.fit_transform(x_train)
x_test = dv.transform(x_test)
# print(dv.get_feature_names())

# 6、决策树估计器流程
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dtc.fit(x_train, y_train)
# 预测
y_predict = dtc.predict(x_test)
# print(y_predict)
score = dtc.score(x_test, y_test)
print(score)

export_graphviz(decision_tree=dtc, out_file="./tree.dot")