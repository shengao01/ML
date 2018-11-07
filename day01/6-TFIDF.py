import jieba


# 结巴分词，将三段话分词之后变成字符串
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def cut_word():
    sentence1 = "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。"
    sentence2 = "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。"
    sentence3 = "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"
    words1 = jieba.cut(sentence1)
    words2 = jieba.cut(sentence2)
    words3 = jieba.cut(sentence3)

    content1 = []
    content2 = []
    content3 = []

    # 把第一句话里面的分词结果放入列表
    for word1 in words1:
        content1.append(word1)

    for word2 in words2:
        content2.append(word2)

    for word3 in words3:
        content3.append(word3)

    return ' '.join(content1), ' '.join(content2), ' '.join(content3)

words1, words2, words3 = cut_word()
# print(words1, words2, words3)
# 特征抽取
tfidf = TfidfVectorizer()
result = tfidf.fit_transform([words1, words2, words3])
# 特征名
print(tfidf.get_feature_names())
# 特征值
print(result.toarray())