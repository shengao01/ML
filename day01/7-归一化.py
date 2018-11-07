from sklearn.preprocessing import MinMaxScaler

data = [[90, 2, 10, 40],
        [60, 4, 15, 45],
        [75, 3, 13, 46]]

# 默认缩放到0-1之间
mms = MinMaxScaler()
# result = mms.fit_transform(data)
# fit + transform
# 从原始数据中获取每一列的最大值最小值
mms.fit(data)
# print(mms.data_min_)
# print(mms.data_max_)
result = mms.transform(data)

print(result)