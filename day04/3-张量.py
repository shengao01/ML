import tensorflow as tf

# a = tf.constant(3) # 0阶张量  标量
# a = tf.constant([1, 1, 1])  # 1阶张量  1维数组
# a = tf.constant([[1, 1, 1], [1, 1, 1]])  # 2阶张量  2维数组

# a = tf.constant(3)
# b = tf.constant(4)
# result = tf.add(a, b)
# # result并没有保存结果，而只是保存的计算的过程
# print(a, b)
# print(result.op)
# # 创建会话，执行Tensor
# sess = tf.Session()
# print(sess.run(result)) # 执行result的op，返回结果
# # 关闭会话
# sess.close()

# a = tf.constant(3.0)
# b = tf.constant(4)
# result = tf.add(a, b)

# 张量的创建
# zeros = tf.zeros(shape=[3, 2], dtype=tf.int32, name="zeros")
# print(zeros)
# ones = tf.ones(shape=[3, 3], dtype=tf.float32)
# 通常初始化参数W取值
# rn = tf.random_normal(shape=[3, 3], dtype=tf.float32)

# 张量的形状
# rn = tf.random_normal(shape=[3, 3], dtype=tf.float32)
# print(rn.get_shape())
# print(rn)
# # 如果张量的形状已经确定，就不可以更改
# # rn.set_shape([3, 3])
# # reshape方法不会改变原来张量的形状，而是创建一个新的张量，设置新的张量的形状
# rn_new = tf.reshape(tensor=rn, shape=[1, 9])
# print(rn_new)


# 张量数据类型的改变
# string_tensor = tf.constant("23")
# number_tensor = tf.string_to_number(string_tensor=string_tensor, out_type=tf.int32)
# print(number_tensor)
# # 将int32转换成float32
# float32_number_tensor = tf.cast(number_tensor, dtype=tf.float32)
#
# sess = tf.Session()
# print(sess.run(float32_number_tensor))
# sess.close()


# 张量的运算
# a = tf.constant(2, dtype=tf.float32)
# result = tf.square(a)
# result = tf.log(x = a)

# 矩阵运算 m x n,  n x k  --> m x k
# a = tf.random_normal(shape=[3, 2])
# b = tf.random_normal(shape=[2, 3])
# # 结果 3 x 3
# result = tf.matmul(a, b)

a = tf.constant([[1, 2, 1], [1, 1, 3]])
# result = tf.reduce_sum(a)
# result = tf.reduce_sum(a, axis=0)
# result = tf.reduce_sum(a, axis=1)
# result = tf.reduce_mean(a, axis=1)
# 每一行最大值的下标
result = tf.argmax(a, axis=1)
sess = tf.Session()
print(sess.run(a))
print(sess.run(result))
sess.close()