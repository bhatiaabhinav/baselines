import os

import joblib
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from baselines import logger
from baselines.a2c.utils import conv, conv_to_fc, fc


class Actor:

    def __init__(self, session: tf.Session, name, ob_shape, ac_shape, softmax_actor=False, nn_size=[64, 64],
                 ob_dtype='float32', q_lr=1e-3, a_lr=1e-4, init_scale=1e-3, use_layer_norm=False,
                 use_batch_norm=False, use_norm_actor=True, l2_reg=1e-2, a_l2_reg=0, clip_norm=None,
                 a_clip_norm=None, tau=0.001, log_transform_action_feed=False, log_transform_max_x=1, log_transform_t=1, **kwargs):
        assert len(ac_shape) == 1
        self.session = session
        self.name = name
        self.ac_shape = ac_shape
        self.tau = tau

        def conv_layers(inputs, inputs_shape, scope, use_ln=False, use_bn=False, training=False):
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

        def hidden_layers(inputs, scope, size, use_ln=False, use_bn=False, training=False):
            with tf.variable_scope(scope):
                for i in range(len(size)):
                    with tf.variable_scope('h{0}'.format(i + 1)):
                        h = fc(inputs, 'fc', size[i], act=lambda x: x)
                        if use_ln:
                            h = tc.layers.layer_norm(
                                h, center=True, scale=True)
                        elif use_bn:
                            h = tf.layers.batch_normalization(
                                h, training=training, name='batch_norm')
                        h = tf.nn.relu(h, name='relu')
                        inputs = h
            return inputs

        def deep_net(inputs, inputs_shape, inputs_dtype, scope, hidden_size, use_ln=False, use_bn=False, training=False, output_size=None):
            conv_needed = len(inputs_shape) > 1
            with tf.variable_scope(scope):
                if conv_needed:
                    inp = tf.divide(tf.cast(inputs, tf.float32, name='cast_to_float'), 255., name='divide_by_255') \
                        if inputs_dtype == 'uint8' else inputs
                    flat = conv_layers(inp, inputs_shape,
                                       'conv_layers', use_ln=use_ln, use_bn=use_bn, training=training)
                else:
                    flat = inputs
                h = hidden_layers(flat, 'hidden_layers',
                                  hidden_size, use_ln=use_ln, use_bn=use_bn, training=training)
                if output_size is not None:
                    final = fc(h, 'output_layer', nh=output_size,
                               act=lambda x: x, init_scale=init_scale)
                else:
                    final = h
                return final

        def safe_softmax(inputs, scope):
            with tf.variable_scope(scope):
                exp = tf.exp(inputs - tf.reduce_max(inputs,
                                                    axis=-1, keep_dims=True, name='max'))
                return exp / tf.reduce_sum(exp, axis=-1, keep_dims=True, name='sum')

        def log_transform(inputs, max_x, t, scope, is_input_normalized=True):
            with tf.name_scope(scope):
                x = max_x * inputs
                return tf.log(1 + x / t) / tf.log(1 + max_x / t)

        with tf.variable_scope(name):
            for scope in ['original', 'target']:
                with tf.variable_scope(scope):
                    states_feed = tf.placeholder(dtype=ob_dtype, shape=[
                                                 None] + list(ob_shape))
                    if scope == 'target':
                        self.states_feed_target = states_feed
                    else:
                        self.states_feed = states_feed
                    with tf.variable_scope('actor'):
                        is_training_a = False if scope == 'target' else tf.placeholder(
                            dtype=tf.bool, name='is_training_a')
                        a = deep_net(states_feed, ob_shape, ob_dtype, 'a_network',
                                     nn_size, use_ln=use_layer_norm and use_norm_actor, use_bn=use_batch_norm and use_norm_actor, training=is_training_a, output_size=ac_shape[0])
                        a = safe_softmax(
                            a, 'softmax') if softmax_actor else tf.nn.tanh(a, 'tanh')
                        use_actions_feed = tf.placeholder(
                            dtype=tf.bool, name='use_actions_feed')
                        actions_feed = tf.placeholder(
                            dtype=tf.float32, shape=[None] + list(ac_shape), name='actions_feed')
                        a = tf.case([
                            (use_actions_feed, lambda: actions_feed)
                        ], default=lambda: a)
                        if scope == 'target':
                            self.a_target = a
                            self.actions_feed_target = actions_feed
                            self.use_actions_feed_target = use_actions_feed
                        else:
                            self.is_training_a = is_training_a
                            self.a = a
                            self.actions_feed = actions_feed
                            self.use_actions_feed = use_actions_feed
                    with tf.variable_scope('critic'):
                        with tf.variable_scope('A'):
                            is_training_critic = False if scope == 'target' else tf.placeholder(
                                dtype=tf.bool, name='is_training_critic')
                            s = deep_net(states_feed, ob_shape, ob_dtype, 'one_hidden',
                                         nn_size[0:1], use_ln=use_layer_norm, use_bn=use_batch_norm, training=is_training_critic)
                            if log_transform_action_feed:
                                a = log_transform(a, log_transform_max_x, log_transform_t, scope='log_transform')
                            s_a_concat = tf.concat(
                                [s, a], axis=-1, name="s_a_concat")
                            A = deep_net(s_a_concat, [nn_size[0] + ac_shape[0]], 'float32', 'A_network',
                                         nn_size[1:], use_ln=use_layer_norm, use_bn=use_batch_norm, training=is_training_critic, output_size=1)[:, 0]
                        V = deep_net(states_feed, ob_shape, ob_dtype, 'V', nn_size, use_ln=use_layer_norm, use_bn=use_batch_norm,
                                     training=is_training_critic, output_size=1)[:, 0]
                        # Q:
                        Q = tf.add(V, A, name='Q')
                        if scope == 'target':
                            self.V_target, self.A_target, self.Q_target = V, A, Q
                        else:
                            self.V, self.A, self.Q = V, A, Q
                            self.is_training_critic = is_training_critic

            with tf.variable_scope('noisy_actor'):
                states_feed = tf.placeholder(dtype=ob_dtype, shape=[
                                             None] + list(ob_shape))
                self.states_feed_noisy_actor = states_feed
                a = deep_net(states_feed, ob_shape, ob_dtype, 'a_network', nn_size, use_ln=use_layer_norm and use_norm_actor,
                             use_bn=use_batch_norm and use_norm_actor, training=False, output_size=ac_shape[0])
                a = safe_softmax(
                    a, 'softmax') if softmax_actor else tf.nn.tanh(a, 'tanh')
                self.a_noisy_actor = a
                self.divergence_noisy_actor = tf.sqrt(
                    tf.reduce_mean(tf.square(self.a - self.a_noisy_actor)))

        # optimizers:
        with tf.name_scope('optimizers'):
            optimizer_A = tf.train.AdamOptimizer(
                learning_rate=q_lr, name='A_adam')
            optimizer_V = tf.train.AdamOptimizer(
                learning_rate=q_lr, name='V_adam')
            optimizer_a = tf.train.AdamOptimizer(
                learning_rate=a_lr, name='actor_adam')

        with tf.name_scope('optimize_actor'):
            # for training actions: maximize Advantage i.e. A
            self.a_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/actor'.format(name))
            self.av_A = tf.reduce_mean(self.A)
            with tf.name_scope('L2_Losses'):
                l2_loss = 0
                for var in self.a_vars:
                    if 'bias' not in var.name:
                        l2_loss += a_l2_reg * tf.nn.l2_loss(var)
            loss = -self.av_A + l2_loss
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope='{0}/original/actor'.format(name))
            with tf.control_dependencies(update_ops):
                a_grads = tf.gradients(loss, self.a_vars)
                if a_clip_norm is not None:
                    a_grads, norm = tf.clip_by_global_norm(
                        a_grads, clip_norm=a_clip_norm)
                self.train_a_op = optimizer_a.apply_gradients(
                    list(zip(a_grads, self.a_vars)))
                # self.train_a_op = optimizer_a.minimize(-self.av_A, var_list=self.a_vars)

        with tf.name_scope('optimize_V'):
            # for training V:
            self.V_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/critic/V'.format(name))
            self.V_target_feed = tf.placeholder(dtype='float32', shape=[None])
            se = tf.square(self.V - self.V_target_feed) / 2
            self.V_mse = tf.reduce_mean(se)
            with tf.name_scope('L2_Losses'):
                l2_loss = 0
                for var in self.V_vars:
                    if 'bias' not in var.name:
                        l2_loss += l2_reg * tf.nn.l2_loss(var)
                loss = self.V_mse + l2_loss
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope='{0}/original/critic/V'.format(name))
            with tf.control_dependencies(update_ops):
                V_grads = tf.gradients(loss, self.V_vars)
                if clip_norm is not None:
                    V_grads, norm = tf.clip_by_global_norm(
                        V_grads, clip_norm=clip_norm)
                self.train_V_op = optimizer_V.apply_gradients(
                    list(zip(V_grads, self.V_vars)))
                # self.train_V_op = optimizer_q.minimize(self.V_mse, var_list=self.V_vars)

        with tf.name_scope('optimize_A'):
            # for training A:
            self.A_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/critic/A'.format(name))
            self.A_target_feed = tf.placeholder(dtype='float32', shape=[None])
            se = tf.square(self.A - self.A_target_feed) / 2
            self.A_mse = tf.reduce_mean(se)
            with tf.name_scope('L2_Losses'):
                l2_loss = 0
                for var in self.A_vars:
                    if 'bias' not in var.name:
                        l2_loss += l2_reg * tf.nn.l2_loss(var)
            loss = self.A_mse + l2_loss
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, scope='{0}/original/critic/A'.format(name))
            with tf.control_dependencies(update_ops):
                A_grads = tf.gradients(loss, self.A_vars)
                if clip_norm is not None:
                    A_grads, norm = tf.clip_by_global_norm(
                        A_grads, clip_norm=clip_norm)
                self.train_A_op = optimizer_A.apply_gradients(
                    list(zip(A_grads, self.A_vars)))
                # self.train_A_op = optimizer_q.minimize(self.A_mse, var_list=self.A_vars)

        with tf.name_scope('target_network_update_ops'):
            # for updating target network:
            from_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, '{0}/original'.format(name))
            to_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, '{0}/target'.format(name))
            self.update_target_network_op, self.soft_update_target_network_op = [], []
            for from_var, to_var in zip(from_vars, to_vars):
                hard_update_op = to_var.assign(from_var)
                soft_update_op = to_var.assign(
                    tau * from_var + (1 - tau) * to_var)
                self.update_target_network_op.append(hard_update_op)
                if 'batch_norm' in from_var.name:
                    soft_update_op = hard_update_op
                self.soft_update_target_network_op.append(soft_update_op)

        with tf.name_scope('noisy_actor_update_ops'):
            from_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/original/actor'.format(name))
            self.params_actor = from_vars
            to_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/noisy_actor'.format(name))
            self.noise_vars = []
            self.update_noisy_actor_op = []
            for from_var, to_var in zip(from_vars, to_vars):
                noise_var = tf.placeholder(
                    shape=from_var.shape.as_list(), dtype=tf.float32)
                self.noise_vars.append(noise_var)
                self.update_noisy_actor_op.append(
                    to_var.assign(from_var + noise_var))

        with tf.name_scope('actor_params_sensitivities'):
            sensitivities_squared = [0] * len(self.params_actor)
            for k in range(ac_shape[0]):
                gradients_k = tf.gradients(self.a[:, k], self.params_actor)
                for var_index in range(len(self.params_actor)):
                    sensitivities_squared[var_index] = sensitivities_squared[var_index] + tf.square(
                        gradients_k[var_index])
            self.actor_params_sensitivities = [
                tf.sqrt(s) for s in sensitivities_squared]

        with tf.name_scope('saving_loading_ops'):
            # for saving and loading
            self.params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, '{0}/original'.format(name))
            self.load_placeholders = []
            self.load_ops = []
            for p in self.params:
                p_placeholder = tf.placeholder(
                    shape=p.shape.as_list(), dtype=tf.float32)
                self.load_placeholders.append(p_placeholder)
                self.load_ops.append(p.assign(p_placeholder))

        # for visualizing computation graph in tensorboard
        self.writer = tf.summary.FileWriter(
            logger.get_dir(), self.session.graph)
        # other summaries:
        # while training V and A:
        self.V_mse_summary = tf.summary.scalar('mb_V_mse', self.V_mse)
        self.V_summary = tf.summary.histogram('mb_V', self.V_target_feed)
        self.A_mse_summary = tf.summary.scalar('mb_A_mse', self.A_mse)
        self.A_summary = tf.summary.histogram('mb_A', self.A_target_feed)
        # while training a:
        self.av_max_A_summary = tf.summary.scalar('mb_av_max_A', self.av_A)
        # while playing:
        self.frame_Q_summary = tf.summary.scalar('frame_Q', self.Q[0])
        self.frame_A_summary = tf.summary.scalar('frame_A', self.A[0])
        # score keeping:
        self.R_placeholder = tf.placeholder(dtype=tf.float32)
        self.R_summary = tf.summary.scalar(
            'Reward_Per_Episode', self.R_placeholder)
        self.R_exploit_placeholder = tf.placeholder(dtype=tf.float32)
        self.R_exploit_summary = tf.summary.scalar(
            'Reward_Per_Episode_Exploit', self.R_exploit_placeholder)
        self.blip_R_exploit_placeholder = tf.placeholder(dtype=tf.float32)
        self.blip_R_exploit_summary = tf.summary.scalar(
            'Blip_Reward_Per_Episode_Exploit', self.blip_R_exploit_placeholder)
        self.score_summary = tf.summary.merge(
            [self.R_summary, self.R_exploit_summary, self.blip_R_exploit_summary])
        # param noise variables:
        self.divergence_placeholder = tf.placeholder(dtype=tf.float32)
        self.divergence_summary = tf.summary.scalar(
            'Param_Noise_Actor_Divergence', self.divergence_placeholder)
        self.adaptive_sigma_placeholder = tf.placeholder(dtype=tf.float32)
        self.adaptive_sigma_summary = tf.summary.scalar(
            'Param_Noise_Adaptive_Sigma', self.adaptive_sigma_placeholder)

    def get_a_V_A_Q(self, states):
        ops, feed = self.get_a_V_A_Q_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)

    def get_a_V_A_Q_op_and_feed(self, states):
        return [self.a, self.V, self.A, self.Q, self.frame_A_summary, self.frame_Q_summary],\
            {self.states_feed: states, self.use_actions_feed: False,
                self.actions_feed: [np.zeros(self.ac_shape)],
                self.is_training_a: False, self.is_training_critic: False}

    def get_noisy_a(self, states):
        return self.session.run(self.a_noisy_actor, feed_dict={self.states_feed_noisy_actor: states})

    def get_divergence_noisy_actor(self, states):
        feed_dict = {
            self.states_feed_noisy_actor: states,
            self.states_feed: states,
            self.is_training_a: False,
            self.use_actions_feed: False,
            self.actions_feed: [np.zeros(self.ac_shape)]
        }
        return self.session.run(self.divergence_noisy_actor, feed_dict=feed_dict)

    def get_V_A_Q(self, states, actions):
        ops, feed = self.get_V_A_Q_op_and_feed(states, actions)
        return self.session.run(ops, feed_dict=feed)

    def get_V_A_Q_op_and_feed(self, states, actions):
        return [self.V, self.A, self.Q, self.frame_A_summary, self.frame_Q_summary],\
            {self.states_feed: states, self.use_actions_feed: True,
                self.actions_feed: actions,
                self.is_training_a: False, self.is_training_critic: False}

    def train_V(self, states, target_V):
        ops, feed = self.get_train_V_op_and_feed(
            states, target_V)
        return self.session.run(ops, feed_dict=feed)

    def get_train_V_op_and_feed(self, states, target_V):
        return [self.train_V_op, self.V_mse, self.V_mse_summary, self.V_summary],\
            {self.states_feed: states, self.V_target_feed: target_V,
                self.is_training_critic: True}

    def train_A(self, states, target_A, actions=None):
        ops, feed = self.get_train_A_op_and_feed(
            states, target_A, actions=actions)
        return self.session.run(ops, feed_dict=feed)

    def get_train_A_op_and_feed(self, states, target_A, actions=None):
        use_actions_feed = actions is not None
        if actions is None:
            actions = [np.zeros(self.ac_shape)]
        return [self.train_A_op, self.A_mse, self.A_mse_summary, self.A_summary],\
            {self.states_feed: states, self.use_actions_feed: use_actions_feed,
                self.actions_feed: actions, self.A_target_feed: target_A,
                self.is_training_a: False, self.is_training_critic: True}

    def train_a(self, states):
        ops, feed = self.get_train_a_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)

    def get_train_a_op_and_feed(self, states):
        return [self.train_a_op, self.av_A, self.av_max_A_summary],\
            {self.states_feed: states, self.use_actions_feed: False,
                self.actions_feed: [np.zeros(self.ac_shape)],
                self.is_training_a: True, self.is_training_critic: False}

    def get_target_a_V_A_Q(self, states):
        ops, feed = self.get_target_a_V_A_Q_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)

    def get_target_a_V_A_Q_op_and_feed(self, states):
        return [self.a_target, self.V_target, self.A_target, self.Q_target], \
            {self.states_feed_target: states,
             self.use_actions_feed_target: False,
             self.actions_feed_target: [np.zeros(self.ac_shape)]}

    def get_target_V_A_Q(self, states, actions):
        ops, feed = self.get_target_V_A_Q_op_and_feed(states, actions)
        return self.session.run(ops, feed_dict=feed)

    def get_target_V_A_Q_op_and_feed(self, states, actions):
        return [self.V_target, self.A_target, self.Q_target], {self.states_feed_target: states, self.use_actions_feed_target: True, self.actions_feed_target: actions}

    def update_target_networks(self):
        ops, feed = self.get_update_target_network_op_and_feed()
        self.session.run(ops, feed_dict=feed)

    def soft_update_target_networks(self):
        ops, feed = self.get_soft_update_target_network_op_and_feed()
        self.session.run(ops, feed_dict=feed)

    def get_update_target_network_op_and_feed(self):
        return [self.update_target_network_op], {}

    def get_soft_update_target_network_op_and_feed(self):
        return [self.soft_update_target_network_op], {}

    def update_noisy_actor(self, param_noise):
        feed_dict = {}
        for noise_var_placeholder, noise_var in zip(self.noise_vars, param_noise):
            feed_dict[noise_var_placeholder] = noise_var
        return self.session.run(self.update_noisy_actor_op, feed_dict=feed_dict)

    def generate_normal_param_noise(self, sigma):
        params = self.session.run(self.params_actor)
        noise = []
        for p in params:
            n = sigma * np.random.standard_normal(size=np.shape(p))
            noise.append(n)
        return noise

    def generate_safe_noise(self, sigma, states):
        feed_dict = {
            self.states_feed: states,
            self.is_training_a: False,
            self.use_actions_feed: False,
            self.actions_feed: [np.zeros(self.ac_shape)]
        }
        params, sensitivities = self.session.run(
            [self.params_actor, self.actor_params_sensitivities], feed_dict=feed_dict)
        noise = []
        for p, s in zip(params, sensitivities):
            s = s / len(states)  # to make s independent of mb size
            n = sigma * np.random.standard_normal(size=np.shape(p))
            n = n / (s + 1e-3)
            noise.append(n)
        return noise

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
