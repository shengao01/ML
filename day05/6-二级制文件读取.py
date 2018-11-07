import os
import tensorflow as tf

RECORD_BYTES = 3073
LABEL_BYTES = 1
IMAGE_BYTES = 3072

# 1、构造二进制文件队列
file_path = "C:/Users/Leon/Desktop/AI/data/cifar10/cifar-10-batches-bin/"
files = os.listdir(file_path)
# print(files)
file_names = [os.path.join(file_path, file) for file in files if file[-3:] == "bin"]
# print(file_names)
queue = tf.train.string_input_producer(file_names)
# 2、构造阅读器，读取二进制数据
reader = tf.FixedLengthRecordReader(record_bytes=RECORD_BYTES)
# key文件名 value 读取的内容
key, value = reader.read(queue)

# 3、解码数据，并分割成label和image（tf.slice）
label_image_raw = tf.decode_raw(bytes=value, out_type=tf.uint8)
label = tf.slice(label_image_raw, begin=[0], size=[LABEL_BYTES])
image = tf.slice(label_image_raw, begin=[LABEL_BYTES], size=[IMAGE_BYTES])
print(label)
print(image)

# 4、图片reshape（对比resize）
image_reshape = tf.reshape(image, shape=[32, 32, 3])

# 6、批处理
label_batch, image_batch = tf.train.batch([label, image_reshape], batch_size=10, num_threads=1, capacity=10)

# 5、开启会话，启动多线程读取
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(image_reshape.eval())
    # print(label_batch.eval())

    coord.request_stop()
    coord.join(threads)



