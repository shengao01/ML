import tensorflow as tf

default_graph = tf.get_default_graph()
print(default_graph)

a = tf.constant(3)
b = tf.constant(4)

print("a所在的计算图：", a.graph)
print("b所在的计算图：", b.graph)

result = tf.add(a, b)
print("result所在的计算图：", result.graph)
print(result)

graph = tf.Graph()
print("新建的计算图:", graph)
# 设置成默认的计算图
with graph.as_default():
    c = tf.constant(5)
    tf.add(a, c)
    print("获取计算图：", tf.get_default_graph())

# 创建会话，执行Tensor
sess = tf.Session()
print("会话所在的计算图：", sess.graph)
print(sess.run(result))
# 关闭会话
sess.close()