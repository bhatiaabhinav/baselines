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


def my_video_schedule(episode_id, total_episodes, video_interval):
    from gym.wrappers.monitor import capped_cubic_video_schedule
    if video_interval is not None and video_interval <= 0:
        return False
    if episode_id == total_episodes - 1:
        return True
    if video_interval is None:
        return capped_cubic_video_schedule(episode_id)
    return episode_id % video_interval == 0


def normalize(a, epsilon=1e-6):
    a = np.clip(a, 0, 1)
    a = a + epsilon
    a = a / np.sum(a)
    return a


def scale(a, low, high, target_low, target_high):
    a_frac = (a - low) / (high - low)
    a = target_low + a_frac * (target_high - target_low)
    return a


def tf_scale(a, low, high, target_low, target_high, scope):
    with tf.variable_scope(scope):
        return scale(a, low, high, target_low, target_high)


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


def tf_log_transform_adaptive(inputs, scope, max_inputs=1, uniform_gamma=False, gamma=None, shift=True, scale=True):
    with tf.variable_scope(scope):
        inputs_shape = inputs.shape.as_list()[1:]
        if gamma is None:
            if uniform_gamma:
                gamma = tf.Variable(1.0, name='gamma', dtype=tf.float32)
            else:
                gamma = tf.Variable(
                    np.ones(inputs_shape, dtype=np.float32), name='gamma')
            gamma = tf.square(gamma, name='gamma_squared')
            # gamma = tf.abs(gamma, name='gamma_abs')
            # gamma = tf.Print(gamma, [gamma])
        epsilon = 1e-3
        log_transform = tf.log(1 + gamma * inputs) / \
            (tf.log(1 + gamma * max_inputs) + epsilon)
        return log_transform


def tf_normalize(inputs, mean, std, scope):
    with tf.variable_scope(scope):
        return (inputs - mean) / std


def tf_safe_softmax(inputs, scope):
    with tf.variable_scope(scope):
        x = inputs - tf.reduce_max(inputs, axis=1, keepdims=True)
        # exp = tf.exp(tf.minimum(inputs, 0))
        exp = tf.exp(x)
        sigma = tf.reduce_sum(exp, axis=-1, keepdims=True, name='sum')
        return exp / sigma
    # return tf_safe_softmax_with_non_uniform_individual_constraints(inputs, [0.1875] * 25, scope)


def tf_safe_softmax_with_uniform_individual_constraints(inputs, max_output, scope):
    with tf.variable_scope(scope):
        dimensions = reduce(mul, inputs.shape.as_list()[1:], 1)
        if max_output < 1 / dimensions or max_output > 1:
            raise ValueError(
                "max_output needs to be in range [1/dimensions, 1]")
        # y = inputs - tf.reduce_max(inputs, axis=1, keepdims=True)
        y = tf.minimum(inputs, 0)
        exp = tf.exp(y)
        sigma = tf.reduce_sum(exp, axis=-1, keepdims=True, name='sum')
        epsilon = dimensions * (1 - max_output) / (dimensions * max_output - 1)
        return (exp + epsilon / dimensions) / (sigma + epsilon)


def tf_safe_softmax_with_non_uniform_individual_constraints(inputs, constraints, scope):
    """adds a max_constrained_softmax layer to compution graph, with as many outputs as inputs

    Arguments:
        inputs {tensor} -- raw (unbounded) output of neural network
        constraints {numpy.ndarray} -- array of max constraints, should of same shape as inputs.
                                        Should sum to more than 1. Each constraint should be in (0, 1].
        scope {string} -- tensorflow name scope for the layer

    Raises:
        ValueError -- if constraints are invalid.

    Returns:
        [tensor] -- s.t. sum of outputs = 1. each output in (0, 1). each output <= corresponding constraint.
    """

    with tf.variable_scope(scope):
        inputs_shape = inputs.shape.as_list()[1:]
        dimensions = reduce(mul, inputs_shape, 1)
        constraints = np.asarray(constraints)
        if list(constraints.shape) != inputs_shape:
            raise ValueError('shape of constraints {0} not compatible with shape of inputs {1}'.format(
                constraints.shape, inputs_shape))
        if np.any(constraints <= 0) or np.any(constraints > 1):
            raise ValueError(
                "constraints need to be in range (0, 1]")
        if np.sum(constraints) <= 1:
            raise ValueError(
                "sum of max constraints needs to be greater than 1")

        # x = inputs - tf.reduce_max(inputs, axis=1, keepdims=True)
        x = tf.minimum(inputs, 0)
        y = tf.exp(x)
        sigma = tf.reduce_sum(y, axis=-1, keepdims=True, name='sum')

        '''
        for some epsilons vector,
        our output z needs to be (y + epsilons)/(sigma + sum(epsilons))
        to satisfy the constraints, we get the following set of linear equations:
        for all i:
            (constraints[i] - 1) * epsilons[i] + constraints[i] * sum(epsilons[except i]) = 1 - constraints[i]
        '''
        constraints_flat = constraints.flatten()
        # to solve the epsilons linear equations: coeffs * epsilons = constants
        # coefficient matrix:
        coeffs = np.array([[(constraints_flat[row] - 1 if col == row else constraints_flat[row])
                            for col in range(dimensions)] for row in range(dimensions)])
        constants = np.array([1 - constraints_flat[row]
                              for row in range(dimensions)])
        epsilons_flat = np.linalg.solve(coeffs, constants)
        epsilons = np.reshape(epsilons_flat, inputs_shape)
        logger.log("constrained_softmax_max: episilons are {0}".format(
            epsilons), level=logger.INFO)
        epsilons_sigma = np.sum(epsilons)
        return (y + epsilons) / (sigma + epsilons_sigma)


def tf_softmax_with_max_constraints(inputs, max_constraints, scope):
    return tf_safe_softmax_with_non_uniform_individual_constraints(inputs, max_constraints, scope)


def tf_softmax_with_dynamic_max_constraints(inputs, constraints, scope):
    with tf.variable_scope(scope):
        inputs_shape = inputs.shape.as_list()[1:]
        dimensions = reduce(mul, inputs_shape, 1)

        x = tf.minimum(inputs, 0)
        y = tf.exp(x)
        sigma = tf.reduce_sum(y, axis=-1, keepdims=True, name='sum')

        '''
        for some epsilons vector,
        our output z needs to be (y + epsilons)/(sigma + sum(epsilons))
        to satisfy the constraints, we get the following set of linear equations:
        for all i:
            (constraints[i] - 1) * epsilons[i] + constraints[i] * sum(epsilons[except i]) = 1 - constraints[i]
        '''
        constraints_flat = tf.reshape(
            constraints, [-1, dimensions], name='constraints_1D')
        # to solve the epsilons linear equations: coeffs * epsilons = constants
        # coefficient matrix:
        coeffs = tf.convert_to_tensor([[(constraints_flat[:, row] - 1 if col == row else constraints_flat[:, row])
                                        for col in range(dimensions)] for row in range(dimensions)], name='coeffs')
        coeffs = tf.transpose(coeffs, [2, 0, 1], name='coeffs_2D')
        constants = tf.convert_to_tensor([1 - constraints_flat[:, row]
                                          for row in range(dimensions)], name='constants')
        constants_1D = tf.transpose(constants, [1, 0], name='constants_1D')
        constants_2D = tf.expand_dims(
            constants_1D, axis=-1, name='constants_2D')
        # epsilons_flat = np.linalg.solve(coeffs, constants)
        coeffs_inverse = tf.matrix_inverse(coeffs, name='coeffs_inverse')
        epsilons = tf.reshape(tf.matmul(coeffs_inverse, constants_2D, name='epsilons_2D'), [-1, dimensions], name='epsilons_1D')
        # epsilons = np.reshape(epsilons_flat, inputs_shape)
        # logger.log("constrained_softmax__dynamic_max: episilons are {0}".format(
        #     epsilons), level=logger.INFO)
        epsilons_sigma = tf.reduce_sum(epsilons, axis=-1, keepdims=True, name='epsilons_sum')
        # epsilons = tf.Print(
        #     epsilons, [epsilons], "constrained_softmax_dynamic_epsilons: ")
        print("epsilons shape", epsilons.shape.as_list())
        return (y + epsilons) / (sigma + epsilons_sigma)


def tf_softmax_with_min_max_constraints(inputs, min_constraints, max_constraints, scope):
    '''we want z_i to sum to 1. s.t. z_i in [m_i, M_i].\n
    so we distribute m_i to z_i first.\n
    then the problem statement becomes:\n
    find vector u s.t. u_i sums to s=1-sum(m_i) and u_i in [0, M_i - m_i]\n
    to do that, we do u = s * max_constrained_softmax(inputs, (M-m)/s)\n
    then z_i = m_i + u_i

    Arguments:
        inputs {tensor} -- raw (unbounded) output of neural network
        min_constraints {numpy.ndarray} -- of same shape as inputs. each component in [0,1). should sum to < 1
        max_constraints {numpy.ndarray} -- of same shape as inputs. each component in (0,1]. should sum to > 1
        scope {str} -- tensorflow name scope for the layer
    '''
    inputs_shape = inputs.shape.as_list()[1:]
    if list(max_constraints.shape) != inputs_shape:
        raise ValueError('shape of max_constraints {0} is not compatible with shape of inputs {1}'.format(
            max_constraints.shape, inputs_shape))
    if list(min_constraints.shape) != inputs_shape:
        raise ValueError('shape of min_constraints {0} is not compatible with shape of inputs {1}'.format(
            min_constraints.shape, inputs_shape))
    if np.any(max_constraints <= 0) or np.any(max_constraints > 1):
        raise ValueError(
            "max_constraints need to be in range (0, 1]")
    if np.any(min_constraints < 0) or np.any(min_constraints >= 1):
        raise ValueError(
            "min_constraints need to be in range [0, 1)")
    if np.any(max_constraints <= min_constraints):
        raise ValueError(
            "max_constraints need to strictly greater than min_constraints")
    if np.sum(max_constraints) <= 1:
        raise ValueError("sum of max_constraints needs to be greater than 1")
    if np.sum(min_constraints) >= 1:
        raise ValueError("sum of min_constraints needs to be less than 1")

    s = 1 - np.sum(min_constraints)
    u_max_constraints = np.minimum(
        1.0, (max_constraints - min_constraints) / s)
    u = s * tf_softmax_with_max_constraints(
        inputs, u_max_constraints, "constrained_softmax_max")
    z = min_constraints + u
    return z


def _tf_nested_softmax_with_min_max_constraints(inputs, constrained_node, scope, z_tree={}, z_node=None):
    '''This algo distributes the value of z_node["tensor"] among its children.
    When called with root_node, assumes that total number of inputs = number of leaf nodes in the constraints_tree\n
    Should be initially called with root_node of constraints tree and empty z_tree and null z_node.
    '''
    with tf.variable_scope(scope):
        # root node case:
        if constrained_node["equals"] is not None:
            assert len(z_tree.keys(
            )) == 0, "An equals_constraint was encountered at a non-root node! Node name: {0}".format(constrained_node["name"])
            assert constrained_node["equals"] == constrained_node["min"] == constrained_node[
                "max"], "At root node, min, max & equal constraints should be same"
            # initialize z_tree:
            z_tree["tensor"] = constrained_node["equals"]
            z_tree["equals"] = constrained_node["equals"]
            z_tree["min"] = constrained_node["min"]
            z_tree["max"] = constrained_node["max"]
            z_tree["name"] = constrained_node["name"]
            z_tree["consumed_inputs"] = 0
            z_node = z_tree

        # base_case: if this the leaf node, return the tensor
        if "children" not in constrained_node or len(constrained_node["children"]) == 0:
            z_node["zone_id"] = constrained_node["zone_id"]
            return [z_node]

        '''
        M = max_constraints of children\n
        m = min_constraints of children\n
        We will distribute min_constraints to the respective nodes first. \n
        The remaining sum will be: \n
            s = z_node["tensor"] - sum(m) \n
        The maximum value of this sum can be:
            S = z_node["max"] - sum(m) \n
        Then we find vector u s.t. sum(u)=s. and u_i in (0, M_i-m_i) \n
            u = s * max_constrained_softmax(inputs, (M-m)/S) \n
        Then: \n
            z_children_i = m_i + u_i \n
        i.e. \n
            z_children_i = m_i + s * max_constrained_softmax(inputs, (M - m)/S)_i \n

        Please convince yourself that z_children_i will be in range (m_i, M_i) and sum(z_children) = z_node["tensor"]
        '''

        children = constrained_node["children"]
        max_constraints = np.array([c["max"] for c in children])
        min_constraints = np.array([c["min"] for c in children])
        assert 0 <= np.sum(min_constraints) <= z_node["min"] <= z_node["max"] < np.sum(
            max_constraints), "The constraints should satisfy 0 <= sum(children_min) <= parent_min <= parent_max < sum(children_max)"
        assert np.all(0 <= min_constraints) and np.all(min_constraints < max_constraints) and np.all(
            max_constraints <= 1), "The constraints should satisfy 0 <= min_constraint_i < max_constraint_i <= 1"

        min_constraints_tensor = tf.constant(min_constraints, dtype=tf.float32, shape=[1, len(children)], name='min_constraints')
        max_constraints_tensor = tf.constant(max_constraints, dtype=tf.float32, shape=[1, len(children)], name='max_constraints')
        s = z_node["tensor"] - tf.reduce_sum(min_constraints_tensor)
        # S = z_node["max"] - sum(min_constraints)
        u_max_constraints = (max_constraints_tensor - min_constraints_tensor) / s
        u_max_constraints = tf.minimum(u_max_constraints, tf.constant(1.0, dtype=tf.float32))
        u = s * tf_softmax_with_dynamic_max_constraints(inputs[:, z_tree["consumed_inputs"]:z_tree["consumed_inputs"] + len(
            children)], u_max_constraints, scope="CSM_on_children_of_{0}".format(z_node["name"]))
        z_tree["consumed_inputs"] += len(children)
        z_children_tensors = np.expand_dims(min_constraints, 0) + u
        z_node["children"] = []
        for i in range(len(children)):
            child = {}
            z_node["children"].append(child)
            child["tensor"] = z_children_tensors[:, i:i + 1]
            child["name"] = children[i]["name"]
            child["min"] = children[i]["min"]
            child["max"] = children[i]["max"]
            child["equals"] = children[i]["equals"]

        return_nodes = []
        for z_node_child, constrained_node_child in zip(z_node["children"], constrained_node["children"]):
            c_returned_nodes = _tf_nested_softmax_with_min_max_constraints(
                inputs, constrained_node_child, z_tree=z_tree, z_node=z_node_child, scope="nested_CSmM_on_children_of_{0}".format(z_node_child["name"]))
            return_nodes.extend(c_returned_nodes)

        return return_nodes


def tf_nested_softmax_with_min_max_constraints(inputs, constraints, scope, z_tree={}):
    with tf.variable_scope(scope):
        leaf_z_nodes = _tf_nested_softmax_with_min_max_constraints(
            inputs, constraints, "_" + scope, z_tree=z_tree)
        leaf_z_tensors = [leaf['tensor'] for leaf in sorted(
            leaf_z_nodes, key=lambda leaf:leaf["zone_id"])]
        return tf.concat(values=leaf_z_tensors, axis=-1, name='concat_leaf_tensors')


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
