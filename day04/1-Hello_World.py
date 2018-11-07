import tensorflow as tf

c = tf.constant("Hello World")
print(c)
# 创建一个会话
sess = tf.Session()
print(sess.run(c))
# 关闭会话
sess.close()