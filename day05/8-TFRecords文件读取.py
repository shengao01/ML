import tensorflow as tf


# 1、构建文件名队列
queue = tf.train.string_input_producer(string_tensor=["C:/Users/Leon/Desktop/AI/ML/day05/tfrecords/cifar10.tfrecords"])
# 2、构造TFRecords阅读器
reader = tf.TFRecordReader()
# 文件名 内容
key, value = reader.read(queue)
# 3、读取解析Example协议数据  返回一个字典类型数据
features = tf.parse_single_example(serialized=value, features={
    "image":tf.FixedLenFeature(shape=[], dtype=tf.string),
    "label":tf.FixedLenFeature(shape=[], dtype=tf.int64)
})
# 4、转换格式，解码图片和标签
image_raw = features["image"]
label = features["label"]
image = tf.decode_raw(image_raw, out_type=tf.uint8)
# 设置图片的形状
image_reshape = tf.reshape(image, shape=[32, 32, 3])

# 6、批处理 image需要reshape，才能进行批处理
label_batch, image_batch = tf.train.batch([label, image_reshape], batch_size=10, num_threads=1, capacity=10)

# 5、开启会话，启动多线程
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(image_reshape.eval())
    print(label.eval())

    coord.request_stop()
    coord.join(threads)


