from sklearn.preprocessing import StandardScaler

data = [[1., -1., 3.],
        [2., 4., 2.],
        [4., 6., -1.]]

ss = StandardScaler()
# 标准化
# result = ss.fit_transform(data)
# fit + transform
# fit计算原始数据平均值和方差
ss.fit(data)
print(ss.mean_)
# 默认转换到均值为0，方差为1
result = ss.transform(data)

print(result)