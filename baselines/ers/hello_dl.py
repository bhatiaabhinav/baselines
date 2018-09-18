import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np

with tf.variable_scope('ournet'):
    hidden_nodes = 10
    x_placeholder = tf.placeholder(dtype='float32', shape=[None, 2], name='x')
    # layer1_weights = tf.Variable(initial_value=np.zeros([2,hidden_nodes]), dtype='float32')
    # layer1_biases = tf.Variable(initial_value=np.zeros(1,hidden_nodes), dtype='float32')
    # h1_sum = tf.matmul(x_placeholder, layer1_weights) + layer1_biases
    # h1 = tf.nn.relu(h1_sum)
    h1 = tf.layers.dense(x_placeholder, hidden_nodes,
                         activation=None, name='h1')
    h1 = tc.layers.layer_norm(h1, scale=True, center=True)
    h1 = tf.nn.relu(h1)
    h2 = tf.layers.dense(h1, hidden_nodes,
                         activation=None, name='h2')
    h2 = tc.layers.layer_norm(h2, scale=True, center=True)
    h2 = tf.nn.relu(h2)
    y = tf.layers.dense(h2, 2)  # this is of shape [None, 2]

variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
print(variables)

with tf.variable_scope('optimization'):
    actual_y_placeholder = tf.placeholder(dtype='float32', shape=[None, 2])
    error_per_data_point = tf.square(y - actual_y_placeholder)
    mse = tf.reduce_mean(error_per_data_point)
    optimizer = tf.train.AdamOptimizer()
    sgd_step = optimizer.minimize(mse, var_list=variables)

variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
print(variables)


def our_underlying_function(x):
    y0 = 100 + np.sin(x[0]) + np.cos(x[1])
    y1 = 100 + np.sin(x[0]) * np.cos(x[1])
    return [y0, y1]


def generate_data(data_size):
    x_data = np.random.sample(size=[data_size, 2])
    x_data = np.pi * (2 * x_data - 1)
    y_data = np.copy(x_data)
    for i in range(data_size):
        y_data[i, :] = our_underlying_function(x_data[i, :])
    return x_data, y_data


def get_mb(x_data, y_data, data_size, mb_size):
    '''gets a random minibatch'''
    random_indices = np.random.choice(list(range(data_size)), size=mb_size)
    x_mb = x_data[random_indices, :]
    y_mb = y_data[random_indices, :]
    return x_mb, y_mb


with tf.Session() as session:
    data_size = 10000
    x_data, y_data = generate_data(data_size)
    x_data_train, y_data_train = x_data[0:7000, :], y_data[0:7000, :]
    x_data_test, y_data_test = x_data[7000:, :], y_data[7000:, :]
    session.run(tf.global_variables_initializer())

    for step_id in range(10000):
        x_mb, y_mb = get_mb(x_data_train, y_data_train, 7000, 32)
        # print(x_mb)
        # print(y_mb)
        _, current_mse = session.run([sgd_step, mse], feed_dict={
            x_placeholder: x_mb,
            actual_y_placeholder: y_mb
        })

        print('Step no: {0}\tmse: {1}'.format(step_id, current_mse))

    test_mse = session.run([mse], feed_dict={
        x_placeholder: x_data_test,
        actual_y_placeholder: y_data_test
    })
    print('Final test mse:', test_mse)
