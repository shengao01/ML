from sklearn.feature_extraction.text import CountVectorizer

data = ["life is short, i like python",
        "life is too long, i dislike python"]

cv = CountVectorizer()
# 特征抽取
# fit_transform = fit + transform
# result = cv.fit_transform(data)

# fit 提取特征
cv.fit(data)
# 特征名
print(cv.get_feature_names())
# transform 数据转换  转换成词频矩阵
result = cv.transform(data)


# 特征值
print(result.toarray())
# 稀疏矩阵
# print(result)
