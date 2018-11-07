from sklearn.feature_selection import VarianceThreshold

data = [[0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3]]

vt = VarianceThreshold()
# 默认过滤掉方差为0特征
result = vt.fit_transform(data)
print(result)
