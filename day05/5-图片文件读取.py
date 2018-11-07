import os
import tensorflow as tf

# 1、构造图片文件队列
file_path = "C:/Users/Leon/Desktop/AI/data/dog/"
files = os.listdir(file_path)
file_names = [os.path.join(file_path, file) for file in files]
# print(file_names)
queue = tf.train.string_input_producer(file_names)
# 2、构造图片阅读器，读取图片数据
reader = tf.WholeFileReader()
# key文件名，value文件的内容
key, value = reader.read(queue)
# 4、解码图片，统一图片大小
image = tf.image.decode_jpeg(value, channels=3)
# 设置图片的大小
image_resize = tf.image.resize_images(image, size=[256, 256])
print(image_resize)

# 批处理
image_batch = tf.train.batch([image_resize], batch_size=10, num_threads=3, capacity=10)
print(image_batch)

# 5、开启会话，启动多线程
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(image_batch.eval())

    coord.request_stop()
    coord.join(threads)

