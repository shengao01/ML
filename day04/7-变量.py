import tensorflow as tf

# var = 3
# print(var)
# var = 4
# print(var)
var = tf.Variable(initial_value=3)
print(var)
# 改变var的值
# var1 = var.assign(4)
# 增加一个delta值，并且赋值给变量，返回一个新的var1
var1 = var.assign_add(delta=1)
# 减小一个delta值，并且赋值给变量，返回一个新的var1
# var.assign_sub(delta=1)

# 得到一个全局变量的初始化器
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 执行初始化器
    sess.run(init)
    print("初始化时的var值", sess.run(var))
    # 执行assign的操作，给变量赋新的值
    sess.run(var1)
    sess.run(var1)
    # print("var1：", sess.run(var1))
    print("更新后的var值", sess.run(var))