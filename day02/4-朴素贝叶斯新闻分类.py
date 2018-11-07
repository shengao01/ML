from sklearn.datasets import fetch_20newsgroups


# 1、加载20类新闻数据，并进行分割
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

news = fetch_20newsgroups(subset="all")
x = news.data
y = news.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
# 2、TF-IDF生成文章特征词
tfidf = TfidfVectorizer()
# 对训练集特征抽取
x_train = tfidf.fit_transform(x_train)
# 测试集按照训练集抽取的特征转换
x_test = tfidf.transform(x_test)
# 3、朴素贝叶斯estimator流程进行预估
mnb = MultinomialNB()
# 训练模型
mnb.fit(x_train, y_train)
# 预测
y_predict = mnb.predict(x_test)
print(y_predict)
# 评估准确率
score = mnb.score(x_test, y_test)
print(score)