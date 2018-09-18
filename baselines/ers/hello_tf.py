import tensorflow as tf
import numpy as np

print("hello tensorflow")

# a = tf.constant(5, name='a')
# b = tf.constant(4, name='b')
# c = tf.add(a, b, name='my_add')
# d = tf.Variable(6, name='d_variable')
# f = tf.placeholder(dtype=tf.int32, shape=None, name='f_placeholder')
# d_assign_op = tf.assign(d, c + f)


# e = c + d

# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     d_init_value = session.run(d)
#     print(d_init_value)
#     c_value, d_value, _ = session.run([c, d, d_assign_op], feed_dict={
#         f: 10
#     })
#     print(c_value, d_value, _)
#     d_final_value = session.run(d)
#     print(d_final_value)

# # print(a,b,c)

with tf.variable_scope('main_graph'):
    _x = tf.Variable([0.0, 0.0], dtype='float32', name='x')
    _x0, _x1 = _x[0], _x[1]
    _f = tf.sin(_x0) + tf.cos(_x0) - tf.square(_x1 - 10.0)

with tf.variable_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    # _grads = tf.gradients(-_f, _x)
    # _gd_step = optimizer.apply_gradients([_grads, _x])
    _gd_step = optimizer.minimize(-_f, var_list=[_x])


with tf.Session() as session:
    writer = tf.summary.FileWriter('/home/abhinav/Desktop/hello_tf', session.graph)
    session.run(tf.global_variables_initializer())
    f = session.run(_f)
    print(f)

    for gd_step_id in range(100):
        _, f = session.run([_gd_step, _f])
        print(f)


        