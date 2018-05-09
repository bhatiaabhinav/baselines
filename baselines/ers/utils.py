import os
import os.path
from functools import reduce
from operator import mul

import joblib
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from keras.initializers import Orthogonal
from keras.layers import Activation, BatchNormalization, Dense
from keras.models import Sequential
from keras.optimizers import Adam

from baselines import logger
from baselines.a2c.utils import conv, conv_to_fc


def normalize(a, epsilon=1e-6):
    a = np.clip(a, 0, 1)
    a = a + epsilon
    a = a / np.sum(a)
    return a


def mutated_ers(alloc, max_mutations=2, mutation_rate=0.05):
    a = alloc.copy()
    for i in range(np.random.randint(1, max_mutations + 1)):
        src = np.random.randint(0, len(alloc))
        dst = np.random.randint(0, len(alloc))
        a[src] -= mutation_rate
        a[dst] += mutation_rate
    return normalize(a)


def mutated_gaussian(alloc, max_mutations=2, mutation_rate=0.05):
    a = alloc.copy()
    for i in range(np.random.randint(1, max_mutations + 1)):
        a = a + np.random.standard_normal(size=np.shape(a))
    a = np.clip(a, -1, 1)
    return a


def tf_log_transform(inputs, max_x, t, scope, is_input_normalized=True):
    with tf.variable_scope(scope):
        x = max_x * inputs
        return tf.log(1 + x / t) / tf.log(1 + max_x / t)


def tf_safe_softmax(inputs, scope):
    with tf.variable_scope(scope):
        exp = tf.exp(tf.minimum(inputs, 0))
        sigma = tf.reduce_sum(exp, axis=-1, keepdims=True, name='sum')
        return exp / sigma


def tf_safe_softmax_with_individual_constraints(inputs, max_output, scope):
    with tf.variable_scope(scope):
        dimensions = reduce(mul, inputs.shape.as_list()[1:], 1)
        if max_output < 1 / dimensions or max_output > 1:
            raise ValueError(
                "max_output needs to be in range [1/dimensions, 1]")
        exp = tf.exp(tf.minimum(inputs, 0))
        sigma = tf.reduce_sum(exp, axis=-1, keepdims=True, name='sum')
        epsilon = dimensions * (1 - max_output) / (dimensions * max_output - 1)
        return (exp + epsilon / dimensions) / (sigma + epsilon)


def tf_conv_layers(inputs, inputs_shape, scope, use_ln=False, use_bn=False, training=False):
    with tf.variable_scope(scope):
        strides = [4, 2, 1]
        nfs = [32, 32, 64]
        rfs = [8, 4, 3]
        num_convs = 3 if inputs_shape[0] >= 60 else 2
        for i in range(3 - num_convs, 3):
            with tf.variable_scope('c{0}'.format(i + 1)):
                c = conv(
                    inputs, 'conv', nfs[i], rfs[i], strides[i], pad='VALID', act=lambda x: x)
                if use_ln:
                    # it is ok to use layer norm since we are using VALID padding. So corner neurons are OK.
                    c = tc.layers.layer_norm(
                        c, center=True, scale=True)
                elif use_bn:
                    c = tf.layers.batch_normalization(
                        c, training=training, name='batch_norm')
                c = tf.nn.relu(c, name='relu')
                inputs = c
        with tf.variable_scope('conv_to_fc'):
            flat = conv_to_fc(inputs)
    return flat


def tf_hidden_layers(inputs, scope, size, use_ln=False, use_bn=False, training=False):
    with tf.variable_scope(scope):
        for i in range(len(size)):
            with tf.variable_scope('h{0}'.format(i + 1)):
                h = tf.layers.dense(inputs, size[i], name='fc')
                if use_ln:
                    h = tc.layers.layer_norm(
                        h, center=True, scale=True)
                elif use_bn:
                    h = tf.layers.batch_normalization(
                        h, training=training, name='batch_norm')
                h = tf.nn.relu(h, name='relu')
                inputs = h
    return inputs


def tf_deep_net(inputs, inputs_shape, inputs_dtype, scope, hidden_size, init_scale=1.0, use_ln=False, use_bn=False, training=False, output_shape=None):
    conv_needed = len(inputs_shape) > 1
    with tf.variable_scope(scope):
        if conv_needed:
            inp = tf.divide(tf.cast(inputs, tf.float32, name='cast_to_float'), 255., name='divide_by_255') \
                if inputs_dtype == 'uint8' else inputs
            flat = tf_conv_layers(inp, inputs_shape,
                                  'conv_layers', use_ln=use_ln, use_bn=use_bn, training=training)
        else:
            flat = inputs
        h = tf_hidden_layers(flat, 'hidden_layers',
                             hidden_size, use_ln=use_ln, use_bn=use_bn, training=training)
        if output_shape is not None:
            output_size = reduce(mul, output_shape, 1)
            # final_flat = tf.layers.dense(h, output_size, kernel_initializer=tf.random_uniform_initializer(
            #     minval=-init_scale, maxval=init_scale), name='output_flat')
            final_flat = tf.layers.dense(h, output_size, kernel_initializer=tf.orthogonal_initializer(
                gain=init_scale), name='output_flat')
            final = tf.reshape(
                final_flat, [-1] + list(output_shape), name='output')
        else:
            final = h
        return final


class Model:
    """Interface for a general ML model"""

    def __init__(self):
        pass

    def predict(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.predict(x)

    def train(self, x, targets):
        raise NotImplementedError()

    def predict_and_test(self, x, actual_y):
        raise NotImplementedError()

    def save(self, save_path):
        raise NotImplementedError()

    def load(self, load_path):
        raise NotImplementedError()


class FFNN_TF(Model):
    def __init__(self, tf_session: tf.Session, scope, input_shape, input_dtype, output_shape, hidden_size, init_scale=1.0, use_ln=False, use_bn=False, lr=1e-3, l2_reg=1e-2, clip_norm=None):
        super().__init__()
        self.session = tf_session
        with tf.variable_scope(scope):
            with tf.variable_scope('network'):
                self.inputs = tf.placeholder(
                    dtype=input_dtype, shape=[None] + input_shape, name='input')
                self.is_training = tf.placeholder(
                    dtype=tf.bool, name='is_training')
                self.outputs = tf_deep_net(
                    self.inputs, input_shape, input_dtype, 'hidden_layers_and_output', hidden_size,
                    output_shape=output_shape, init_scale=init_scale,
                    use_ln=use_ln, use_bn=use_bn, training=self.is_training)
            with tf.variable_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            with tf.variable_scope('optimization_ops'):
                self.trainable_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                self.targets = tf.placeholder(
                    dtype='float32', shape=[None] + output_shape)
                se = tf.square(self.outputs - self.targets)
                self.mse = tf.reduce_mean(se)
                with tf.variable_scope('L2_Losses'):
                    l2_loss = 0
                    for var in self.trainable_vars:
                        if 'bias' not in var.name and 'output' not in var.name:
                            l2_loss += l2_reg * tf.nn.l2_loss(var)
                    self.loss = self.mse + l2_loss
                update_ops = tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS, scope=scope)
                with tf.control_dependencies(update_ops):
                    self.grads = tf.gradients(self.loss, self.trainable_vars)
                    if clip_norm is not None:
                        self.grads = [tf.clip_by_norm(
                            grad, clip_norm=clip_norm) for grad in self.grads]
                    self.train_op = self.optimizer.apply_gradients(
                        list(zip(self.grads, self.trainable_vars)))

            with tf.variable_scope('saving_loading_ops'):
                # for saving and loading
                self.params = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                self.load_placeholders = []
                self.load_ops = []
                for p in self.params:
                    p_placeholder = tf.placeholder(
                        shape=p.shape.as_list(), dtype=tf.float32)
                    self.load_placeholders.append(p_placeholder)
                    self.load_ops.append(p.assign(p_placeholder))

            self.writer = tf.summary.FileWriter(
                logger.get_dir(), self.session.graph)

    def predict(self, x):
        return self.session.run(self.outputs, feed_dict={
            self.inputs: x,
            self.is_training: False
        })

    def __call__(self, x):
        return self.predict(x)

    def train(self, x, targets):
        '''returns mse, loss'''
        return self.session.run([self.train_op, self.mse, self.loss], feed_dict={
            self.inputs: x,
            self.targets: targets,
            self.is_training: True
        })[1:]

    def predict_and_test(self, x, actual_y):
        '''returns predicted y and mse against actual y'''
        return self.session.run([self.outputs, self.mse], feed_dict={
            self.inputs: x,
            self.is_training: False,
            self.targets: actual_y
        })

    def save(self, save_path):
        params = self.session.run(self.params)
        from baselines.a2c.utils import make_path
        make_path(os.path.dirname(save_path))
        joblib.dump(params, save_path)
        self.writer.flush()

    def load(self, load_path):
        params = joblib.load(load_path)
        feed_dict = {}
        for p, p_placeholder in zip(params, self.load_placeholders):
            feed_dict[p_placeholder] = p
        self.session.run(self.load_ops, feed_dict=feed_dict)


class FFNN_Keras(Model):
    def __init__(self, session, scope, input_shape, input_dtype, output_shape, hidden_size, init_scale=1.0, use_ln=False, use_bn=False, lr=1e-3, l2_reg=1e-2, clip_norm=None):
        super().__init__()
        if l2_reg > 0:
            raise NotImplementedError()
        if clip_norm is not None:
            raise NotImplementedError()

        model = Sequential(name=scope)
        conv_needed = len(input_shape) > 1
        if conv_needed:
            raise NotImplementedError()
        for hs in hidden_size:
            model.add(
                Dense(units=hs, input_shape=input_shape, kernel_initializer=Orthogonal(seed=0)))
            if use_ln:
                raise NotImplementedError()
            if use_bn:
                model.add(BatchNormalization())
            model.add(Activation('relu'))
        output_size = reduce(mul, output_shape, 1)
        model.add(Dense(units=output_size,
                        kernel_initializer=Orthogonal(gain=init_scale, seed=0)))
        if len(output_shape) > 1:
            raise NotImplementedError()
        optimizer = Adam(lr=lr)
        model.compile(loss='mse', optimizer=optimizer)
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def train(self, x, targets):
        return [self.model.train_on_batch(x, targets), 0]

    def predict_and_test(self, x, actual_y):
        pred = self.model.predict(x)
        mse = self.model.evaluate(x, actual_y)
        return [pred, mse]

    def save(self, save_path):
        self.model.save_weights(save_path)

    def load(self, load_path):
        self.model.load_weights(load_path)


def StaffordRandFixedSum(n, u, nsets):

    # deal with n=1 case
    if n == 1:
        return np.tile(np.array([u]), [nsets, 1])

    k = np.floor(u)
    s = u
    step = 1 if k < (k - n + 1) else -1
    s1 = s - np.arange(k, (k - n + 1) + step, step)
    step = 1 if (k + n) < (k - n + 1) else -1
    s2 = np.arange((k + n), (k + 1) + step, step) - s

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, (n + 1)):
        tmp1 = w[i - 2, np.arange(1, (i + 1))] * \
            s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * \
            s2[np.arange((n - i), n)] / float(i)
        w[i - 1, np.arange(1, (i + 1))] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, (i + 1))] + tiny
        tmp4 = np.array(
            (s2[np.arange((n - i), n)] > s1[np.arange(0, i)]))
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + \
            (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

    m = nsets
    x = np.zeros((n, m))
    rt = np.random.uniform(size=(n - 1, m))  # rand simplex type
    rs = np.random.uniform(size=(n - 1, m))  # rand position in simplex
    s = np.repeat(s, m)
    j = np.repeat(int(k + 1), m)
    sm = np.repeat(0, m)
    pr = np.repeat(1, m)

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0)
        e = (rt[(n - i) - 1, ...] <= t[i - 1, j - 1])
        sx = rs[(n - i) - 1, ...] ** (1 / float(i))  # next simplex coord
        sm = sm + (1 - sx) * pr * s / float(i + 1)
        pr = sx * pr
        x[(n - i) - 1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1, ...] = sm + pr * s

    # iterated in fixed dimension order but needs to be randomised
    # permute x row order within each column
    for i in range(0, m):
        x[..., i] = x[np.random.permutation(n), i]

    return np.transpose(x)
