import tensorflow as tf

# a = tf.placeholder(dtype=tf.int32)
# b = tf.placeholder(dtype=tf.int32)
# result = tf.add(a, b)
# print(result)
# # 创建会话，执行Tensor
# sess = tf.Session()
# # print(sess.run(result, feed_dict={a: 3, b: 4}))
# print(result.eval(session=sess, feed_dict={a: 3, b: 4}))
# # 关闭会话
# sess.close()

a = tf.placeholder(dtype=tf.int32)
# 由于占位符没有指定形状，后面可以更改它的形状
a.set_shape([3, 2])
print(a.get_shape())
# 如果placeholder形状已经设置好了就不能再更改
# a.set_shape([2, 3])