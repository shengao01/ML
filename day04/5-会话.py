import tensorflow as tf

a = tf.constant(3)
b = tf.constant(4)
result = tf.add(a, b)
print(result)
# 创建会话，执行Tensor
# sess = tf.Session()
# # print(sess.run(result))
# 指定会话
# print(result.eval(session=sess))
# # 关闭会话，会话管理者计算的资源--CPU、GPU, 释放资源
# sess.close()

# 获取默认会话, Tensorflow中有默认的计算图，但是没有默认的会话
session = tf.get_default_session()
print(session)

# 创建默认的计算图
# allow_soft_placement,如果是使用GPU版本，在GPU不可用的情况下，自动切换到CPU， 建议使用GPU版本一般打开
# log_device_placement 开启log，描述计算在什么样的设备执行, 在生产环境，关闭
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess = tf.Session(config=config)
with sess.as_default():
    # sess = tf.get_default_session()
    print(result.eval())

# 使用上下文管理器自动释放资源
with tf.Session() as sess:
    print(sess.run(result))
    # 不用指定会话
    print(result.eval())