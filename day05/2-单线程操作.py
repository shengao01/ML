import tensorflow as tf

queue = tf.FIFOQueue(capacity=3, dtypes=tf.int32)
c = tf.constant([1, 2, 3])
init_queue = queue.enqueue_many(c)

# 出队， + 1 ， 入队
var = queue.dequeue()
# var_add = tf.add(var, tf.constant(1))
# var_add = tf.add(var, 1)
var_add = var + 1
enqueue_op = queue.enqueue(var_add)

with tf.Session() as sess:
    sess.run(init_queue)
    # 出队， + 1 ， 入队
    sess.run(enqueue_op)
    for i in range(3):
        print(queue.dequeue().eval())

