import os
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


tf.flags.DEFINE_integer(flag_name="steps", default_value=1000, docstring="训练的次数")
FLAGS = tf.flags.FLAGS

def main(argv):
    steps = FLAGS.steps
    print("获取到自定义参数steps:", steps)

    # 定义输入
    with tf.variable_scope("input"):
        x = tf.placeholder(dtype=tf.float32, name="x")
        y = tf.placeholder(dtype=tf.float32, name="y")

    with tf.variable_scope("model"):
        # 定义参数
        weight = tf.Variable(initial_value=3.0, name="weight")
        bias = tf.Variable(initial_value=3.0, name="bias")
        # 构建模型  y = 1 * x + 1
        linear_model = tf.multiply(weight, x) + bias
        # tf.add(weight * x, bias)

    with tf.variable_scope("loss"):
        # 损失函数
        loss = tf.reduce_sum(tf.square(linear_model - y))

    # 调整参数使得损失最少
    # weight_update = weight.assign(1)
    # bias_update = bias.assign(1)
    with tf.variable_scope("train"):
        train_op = GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    file_writer = tf.summary.FileWriter(logdir="./log", graph=tf.get_default_graph())
    # 收集weight bias loss 的变化
    tf.summary.scalar(tensor=weight, name="weight")
    tf.summary.scalar(tensor=bias, name="bias")
    tf.summary.scalar(tensor=loss, name="loss")

    # 定义合并操作
    merge_all = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    # 定义Saver保存模型
    saver = tf.train.Saver(var_list=[weight, bias], max_to_keep=3)

    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 执行assign，更新参数
        # sess.run([weight_update, bias_update])
        # print(sess.run(loss, feed_dict={x:[0, 1, 2, 3], y:[1, 2, 3, 4]}))

        # 检查是否有checkpoint文件，如果有，加载里面保存的参数到会话
        if os.path.exists("./model/checkpoint"):
            saver.restore(sess=sess, save_path="./model/") # 路径是目录，最后加/

        for i in range(steps):
            sess.run(train_op, feed_dict={x:[0, 1, 2, 3], y:[1, 2, 3, 4]}) # 执行训练操作
            # 每次训练之后会自动更新参数weight, bias
            print(sess.run([weight, bias]))
            # 每次迭代， 执行合并操作
            summary = sess.run(merge_all, feed_dict={x: [0, 1, 2, 3], y: [1, 2, 3, 4]})
            file_writer.add_summary(summary, i)

            # if i % 100 ==0 :
            #     saver.save(sess=sess, save_path="./model/")
        saver.save(sess=sess, save_path="./model/")

if __name__ =="__main__":
    tf.app.run()