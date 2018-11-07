from sklearn.feature_extraction import DictVectorizer

data = [{'city': '北京','temperature':100},
{'city': '上海','temperature':60},
{'city': '深圳','temperature':30}]

dv = DictVectorizer(sparse=False)
result = dv.fit_transform(data)
# 特征名
print(dv.get_feature_names())
# 特征值
print(result)
