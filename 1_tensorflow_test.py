import tensorflow as tf


con1 = tf.constant("hello world")

renwu = tf.Session()

print(con1.op)
print("====================")
print(renwu.run(con1))

renwu.close()