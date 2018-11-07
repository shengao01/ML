import tensorflow as tf

queue = tf.FIFOQueue(capacity=100, dtypes=tf.int32)

# 实现变量加1，入队
var = tf.Variable(initial_value=0, dtype=tf.int32)
var_add = var.assign_add(delta=1)
enqueue_op = queue.enqueue(var_add)

# 创建队列管理器
queue_runner = tf.train.QueueRunner(queue=queue, enqueue_ops=[enqueue_op]*1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 线程协调器
    coord = tf.train.Coordinator()
    # 创建线程，并且启动，这些线程就会开始执行入队操作
    threads = queue_runner.create_threads(sess=sess, coord=coord, start=True)
    # 队列数据的入队操作由queue runner的线程去完成， 只需要在主线程出队操作
    queue.dequeue().eval()
    for i in range(100):
        print(queue.size().eval())

    # 请求线程终止
    coord.request_stop()
    # 等待线程终止
    coord.join(threads)