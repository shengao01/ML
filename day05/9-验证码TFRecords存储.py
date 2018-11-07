import os
import tensorflow as tf

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 1、读取标签csv文件内容 文件名不能随机出队
def read_csv():
    queue = tf.train.string_input_producer(string_tensor=["C:/Users/Leon/Desktop/AI/data/GenPics/labels.csv"], shuffle=False)
    # 阅读器
    reader = tf.TextLineReader()
    key, value = reader.read(queue)
    # 解码一行数据
    index, label = tf.decode_csv(records=value, record_defaults=[[0],["None"]])
    label_batch = tf.train.batch([label], batch_size=6000, num_threads=1, capacity=6000)
    return label_batch


# 2、将字符串转换成数字形式
def transform_label_to_digist(label_batch):
    # {0: A, 1:B, ...............}
    num_letter = dict(enumerate(list(LETTERS)))
    # 键值反转
    letter_num = dict(zip(num_letter.values(), num_letter.keys()))
    # 构建一个字典数据 {A: 0, B: 1, C：2 ......N : 13}

    # [b'NZPP' b'WKHK' b'WPSJ'..., b'FVQJ' b'BQYA' b'BCHR']
    # 遍历6000
    result = []
    for i in  range(6000):
        # NZPP对应的数字列表
        label_digits = []
        # 遍历每一个字母
        for letter in label_batch[i].decode("utf-8"):
            # 获取到一个字母对应的数字，加入到label_digits
            label_digits.append(letter_num[letter])
        result.append(label_digits)
    return result


# 3、读取验证码图片
def read_captcha():
    file_path = "C:/Users/Leon/Desktop/AI/data/GenPics/"
    # 不能使用os.listdir获取图片列表，不是按顺序
    # os.listdir(file_path)
    file_names = []
    for i in range(6000):
        file = str(i) + ".jpg"
        file_names.append(os.path.join(file_path, file))
    # print(file_names
    # 创建文件名队列, 不能随机出队
    queue= tf.train.string_input_producer(file_names, shuffle=False)
    reader = tf.WholeFileReader()
    key, value = reader.read(queue)
    image = tf.image.decode_jpeg(contents=value, channels=3)
    # 设置形状
    image_reshape = tf.reshape(image, [20, 80, 3])
    image_batch = tf.train.batch([image_reshape], num_threads=1, batch_size=6000, capacity=6000)
    return image_batch


# 4、将验证码和标签存储到TFRecords
def wirte_to_tfrecords(label_batch, image_batch):
    writer = tf.python_io.TFRecordWriter("./tfrecords/captha.tfrecords")
    # 遍历10张图片和10个标签，构建example协议块，存入tfrecords中
    label_batch = tf.cast(label_batch, tf.uint8) # 将标签数据转换成6000Tensor
    for i in range(10):
        image = image_batch[i].tostring()
        label = label_batch[i].eval().tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "label":tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
        }))
        writer.write(example.SerializeToString())
    writer.close()



label_batch = read_csv()
image_batch = read_captcha()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    # 启动多线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # [b'NZPP' b'WKHK' b'WPSJ'..., b'FVQJ' b'BQYA' b'BCHR']
    # print(label_batch.eval())
    # [[13, 25, 15, 15], [22, 10, 7, 10], [22, 15, 18, 9], [16, 6, 13, 10], [1, 0, 8, 17], [0, 9, 24, 14], [7, 10, 17, 25], [10, 20, 0, 18].....
    label_disigts = transform_label_to_digist(label_batch.eval())
    # print(label_disigts)

    print(image_batch.eval())
    # wirte_to_tfrecords(label_disigts, image_batch.eval())

    coord.request_stop()
    coord.join(threads)
