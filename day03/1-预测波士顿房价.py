

# 1、波士顿地区房价数据获取
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

boston = load_boston()
# 2、波士顿地区房价数据分割
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target)
# 3、训练与测试数据标准化处理
ss = StandardScaler()
# 对训练集的特征值转换
x_train = ss.fit_transform(x_train)
# 对测试集特征值的转换
x_test = ss.transform(x_test)
# 解决回归问题，如果对特征值做了标准化，同样对对目标值做标准化
ss_y = StandardScaler()
# 对训练集的目标值的标准化，不能使用原来对特征值的StandardScaler，
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))

# 4、使用最简单的线性回归模型LinearRegression和
# 梯度下降估计SGDRegressor对房价进行预测
# 使用正规方程一次求解回归系数
lr = LinearRegression()
# 完成训练
lr.fit(x_train, y_train)
# 预测的房价
lr_predict = lr.predict(x_test)
# 由于使用x_test已经标准化了，得到结果相当于标准化结果，需要进行逆标准化，才能得到最终的结果
# print("正规方程预测的结果：",ss_y.inverse_transform(lr_predict))
# print("正规方程得到的回归系数：", lr.coef_)
# 均方误差
lr_error = mean_squared_error(y_true=y_test, y_pred=ss_y.inverse_transform(lr_predict))
print("正则方程的均方误差：", lr_error)

sgd = SGDRegressor()
sgd.fit(x_train, y_train)
sgd_predict = sgd.predict(x_test)
# print("梯度下降预测的结果：", ss_y.inverse_transform(sgd_predict))
# print("梯度下降的回归系数：", sgd.coef_)
sgd_error = mean_squared_error(y_true=y_test, y_pred=ss_y.inverse_transform(sgd_predict))
print("梯度下降的均方误差：", sgd_error)

# 岭回归
# ridge = Ridge(alpha=3)
# ridge.fit(x_train, y_train)
# ridge_predict = ridge.predict(x_test)
# ridge_error = mean_squared_error(y_true=y_test, y_pred=ss_y.inverse_transform(ridge_predict))
# print("岭回归的均方误差：", ridge_error)

# Lasso回归
# lasso = Lasso(alpha=0.01)
# lasso.fit(x_train, y_train)
# lasso_predict = lasso.predict(x_test)
# lasso_error = mean_squared_error(y_true=y_test, y_pred=ss_y.inverse_transform(lasso_predict))
# print("Lasso回归的均方误差：", lasso_error)

alphas = [0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100]
ridge = RidgeCV(alphas=alphas)
ridge.fit(x_train, y_train)
print("岭回归最优的正则化力度：", ridge.alpha_)


lasso = LassoCV(alphas=alphas)
lasso.fit(x_train, y_train)
print("Lasso回归最优化的正则化力度：", lasso.alpha_)