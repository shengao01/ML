import os
import tensorflow as tf

# 1、构建文件名（绝对路径）队列
file_path = "C:/Users/Leon/Desktop/AI/data/csvdata/"
files = os.listdir(file_path)
# print(files)
# 构建绝对路径的列表
file_names = [os.path.join(file_path, file) for file in files]
# print(file_names)
# string_input_producer返回一个队列，并且创建队列对应的QueueRunner，将这个QueueRunner加入计算图的QueueRunner集合
queue = tf.train.string_input_producer(file_names)
# 2、构建文件阅读器，读取内容
reader = tf.TextLineReader()
# key就是读取的文件名， value默认一行的内容
key, value = reader.read(queue)
# 3、解码内容，指定默认值或者缺省值
records_default = [["None"], ["None"]]
# 读取的一行数据有两列
value1, value2 = tf.decode_csv(records=value, record_defaults=records_default)

# 批处理
value1_batch, value2_batch = tf.train.batch(tensors=[value1, value2], batch_size=9, num_threads=1, capacity=9)

# 4、开启会话，启动队列管理器
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # Starts all queue runners collected in the graph.
    # 启动所有在计算图内QueueRunner集合的QueueRunner
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # .....
    print([value1_batch.eval(), value2_batch.eval()])

    coord.request_stop()
    coord.join(threads)


