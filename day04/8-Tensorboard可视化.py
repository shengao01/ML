import tensorflow as tf

a = tf.constant(3, name="a")
b = tf.constant(4, name="b")
result = tf.add(a, b)
print(result)
# 创建会话，执行Tensor
sess = tf.Session()
print(sess.run(result))

# 创建FileWriter, 序列化成events文件, 注意一下不要写成"./log/"
tf.summary.FileWriter(logdir="./log", graph=tf.get_default_graph())

# 关闭会话
sess.close()

