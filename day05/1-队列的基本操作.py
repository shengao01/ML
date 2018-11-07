import tensorflow as tf

queue = tf.FIFOQueue(capacity=2, dtypes=tf.int32)

c = tf.constant(1)
enqueue_op = queue.enqueue(vals=c)


# c = tf.constant([1, 2, 3])
# enqueue_op = queue.enqueue_many(c)

# print(enqueue_op)
out = queue.dequeue()
print(out)
with tf.Session() as sess:
    print("入队之前队列大小:", queue.size().eval())
    sess.run(enqueue_op)
    print("入队之后队列大小:", queue.size().eval())
    print("出队：", out.eval())
    print("出队：", out.eval())
