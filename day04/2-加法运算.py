import tensorflow as tf

# def add(a, b):
#     return a + b
#
# a = 3
# b = 4
# print(add(a, b))

a = tf.constant(3)
b = tf.constant(4)
result = tf.add(a, b)
print(result)
# 创建会话，执行Tensor
sess = tf.Session()
print(sess.run(result))
# 关闭会话
sess.close()