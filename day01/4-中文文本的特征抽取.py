import jieba
from sklearn.feature_extraction.text import CountVectorizer

# 单个字或者字母不当作特征
data = ["生活 很短，我 喜欢 python", "生活 太久了，我 不喜欢 python"]
cv = CountVectorizer()
# 特征抽取
# result = cv.fit_transform(data)

# 特征名
# print(cv.get_feature_names())
# 特征值
# print(result.toarray())

result = jieba.cut("我是一个好程序员")

# 遍历分词结果，加入列表
content = []
for word in result:
    content.append(word)

print(' '.join(content))