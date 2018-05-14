import os

import joblib
import numpy as np
import tensorflow as tf
from gym.spaces import Box

from baselines import logger
from baselines.ers.utils import (tf_deep_net, tf_log_transform_adaptive,
                                 tf_safe_softmax_with_non_uniform_individual_constraints)


class DDPG_Model_Base:
    def __init__(self, session: tf.Session, name, ob_space: Box, ac_space: Box, softmax_actor, nn_size, init_scale, advantage_learning, use_layer_norm, use_batch_norm, use_norm_actor, log_transform_inputs, **kwargs):
        assert len(
            ac_space.shape) == 1, "Right now only flat action spaces are supported"
        self.session = session
        self.name = name
        self.ac_shape = ac_space.shape
        self.ac_high = ac_space.high
        self.ob_shape = ob_space.shape
        self.ob_dtype = ob_space.dtype
        self.ob_high = ob_space.high
        self.nn_size = nn_size
        self.init_scale = init_scale
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_norm_actor = use_norm_actor
        self.log_transform_inputs = log_transform_inputs
        # self.log_transform_action_feed = log_transform_action_feed
        # self.log_transform_max_x = log_transform_max_x
        # self.log_transform_t = log_transform_t
        self.softmax_actor = softmax_actor
        self.advantage_learning = advantage_learning
        self.DUMMY_ACTION = [np.zeros(self.ac_shape)]

    def _setup_states_feed(self):
        self._states_feed = tf.placeholder(dtype=self.ob_dtype, shape=[
                                           None] + list(self.ob_shape), name="states_feed")

    def _setup_actor(self):
        with tf.variable_scope('model/actor'):
            self._is_training_a = tf.placeholder(
                dtype=tf.bool, name='is_training_a')
            states = self._tf_normalize_states(
                self._states_feed, 'normalized_states', self._is_training_a)
            a = tf_deep_net(states, self.ob_shape, self.ob_dtype, 'a_network', self.nn_size, use_ln=self.use_layer_norm and self.use_norm_actor,
                            use_bn=self.use_batch_norm and self.use_norm_actor, training=self._is_training_a, output_shape=self.ac_shape)
            if self.softmax_actor:
                a = tf_safe_softmax_with_non_uniform_individual_constraints(
                    a, self.ac_high, 'constrained_softmax')
            else:
                a = tf.nn.tanh(a, 'tanh')
            self._use_actions_feed = tf.placeholder(
                dtype=tf.bool, name='use_actions_feed')
            self._actions_feed = tf.placeholder(
                dtype=tf.float32, shape=[None] + list(self.ac_shape), name='actions_feed')
            self._a = tf.case([
                (self._use_actions_feed, lambda: self._actions_feed)
            ], default=lambda: a)

    def _tf_normalize_states(self, states, scope, is_training):
        with tf.variable_scope(scope):
            if self.log_transform_inputs:
                zones = self.ac_shape[0]
                states_feed_demand = states[:, :-zones - 1]
                states_feed_alloc = states[:, -zones - 1:-1]
                states_feed_time = states[:, -1:]
                # states_feed_demand = tf.layers.batch_normalization(
                #     states_feed_demand, training=is_training, name='batch_norm_demand')
                states_feed_demand = tf_log_transform_adaptive(
                    states_feed_demand, 'log_transform_demand', max_inputs=self.ob_high[:-zones - 1], uniform_beta=True)
                # states_feed_alloc = tf.layers.batch_normalization(
                #     states_feed_alloc, training=is_training, name='batch_norm_alloc')
                states_feed_alloc = tf_log_transform_adaptive(
                    states_feed_alloc, 'log_transform_alloc', max_inputs=self.ob_high[-zones - 1:-1], uniform_beta=True)
                states = tf.concat(
                    [states_feed_demand, states_feed_alloc, states_feed_time], axis=-1, name='states_concat')
                # states = tf.layers.batch_normalization(
                #     states, training=is_training, name='batch_norm')
            else:
                states = tf.layers.batch_normalization(
                    states, training=is_training, name='batch_norm')
            return states

    def _setup_critic(self):
        with tf.variable_scope('model/critic'):
            self._is_training_critic = tf.placeholder(
                dtype=tf.bool, name='is_training_critic')
            with tf.variable_scope('A'):
                states = self._tf_normalize_states(
                    self._states_feed, 'normalized_states', self._is_training_critic)
                s_after_one_hidden = tf_deep_net(states, self.ob_shape, self.ob_dtype, 'one_hidden',
                                                 self.nn_size[0:1], use_ln=self.use_layer_norm, use_bn=self.use_batch_norm, training=self._is_training_critic, output_shape=None)
                if self.log_transform_inputs:
                    a = tf_log_transform_adaptive(
                        self._a, scope='log_transform', uniform_beta=True, max_inputs=self.ac_high)
                else:
                    a = tf.layers.batch_normalization(
                        self._a, training=self._is_training_critic, name='batch_norm')
                s_a_concat = tf.concat(
                    [s_after_one_hidden, a], axis=-1, name="s_a_concat")
                self._A = tf_deep_net(s_a_concat, [self.nn_size[0] + self.ac_shape[0]], 'float32', 'A_network',
                                      self.nn_size[1:], use_ln=self.use_layer_norm, use_bn=self.use_batch_norm, training=self._is_training_critic, output_shape=[1])[:, 0]
            if self.advantage_learning:
                states = self._tf_normalize_states(
                    self._states_feed, 'normalized_states', self._is_training_critic)
                self._V = tf_deep_net(states, self.ob_shape, self.ob_dtype, 'V', self.nn_size, use_ln=self.use_layer_norm,
                                      use_bn=self.use_batch_norm, training=self._is_training_critic, output_shape=[1])[:, 0]
                self._Q = tf.add(self._V, self._A, name='Q')
            else:
                self._Q = self._A

    def _get_tf_variables(self, extra_scope=''):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '{0}/model/{1}'.format(self.name, extra_scope))

    def _get_tf_trainable_variables(self, extra_scope=''):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/model/{1}'.format(self.name, extra_scope))

    def _get_tf_perturbable_variables(self, extra_scope=''):
        return [var for var in self._get_tf_variables(extra_scope) if not('LayerNorm' in var.name or 'batch_norm' in var.name or 'log_transform' in var.name)]

    def _get_update_ops(self, extra_scope=''):
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='{0}/model/{1}'.format(self.name, extra_scope))

    def get_a(self, states):
        return self.session.run(self._a, feed_dict={
            self._states_feed: states,
            self._is_training_a: False,
            self._use_actions_feed: False,
            self._actions_feed: self.DUMMY_ACTION,
        })

    def get_a_V_A_Q(self, states):
        return self.session.run([self._a, self._V, self._A, self._Q], feed_dict={
            self._states_feed: states,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: False,
            self._use_actions_feed: False,
            self._is_training_critic: False
        })

    def get_a_Q(self, states):
        return self.session.run([self._a, self._Q], feed_dict={
            self._states_feed: states,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: False,
            self._use_actions_feed: False,
            self._is_training_critic: False
        })

    def get_V_A_Q(self, states, actions):
        return self.session.run([self._V, self._A, self._Q], feed_dict={
            self._states_feed: states,
            self._use_actions_feed: True,
            self._actions_feed: actions,
            self._is_training_a: False,
            self._is_training_critic: False
        })

    def get_Q(self, states, actions):
        return self.session.run([self._Q], feed_dict={
            self._states_feed: states,
            self._use_actions_feed: True,
            self._actions_feed: actions,
            self._is_training_a: False,
            self._is_training_critic: False
        })

    def Q(self, s, a):
        return self.get_Q(s, a)

    def max_Q(self, s):
        return self.get_a_Q(s)[1]

    def V(self, s):
        return self.get_a_V_A_Q(s)[1]

    def A(self, s, a):
        return self.get_V_A_Q(s, a)[1]

    def max_A(self, s):
        return self.get_a_V_A_Q(s)[2]

    def argmax_Q(self, s):
        return self.get_a(s)


class DDPG_Model_Main(DDPG_Model_Base):
    def __init__(self, session: tf.Session, name, ob_space: Box, ac_space: Box, softmax_actor, nn_size,
                 lr, a_lr, init_scale, advantage_learning, use_layer_norm,
                 use_batch_norm, use_norm_actor, l2_reg, a_l2_reg, clip_norm,
                 a_clip_norm, log_transform_inputs, **kwargs):
        super().__init__(session=session, name=name, ob_space=ob_space, ac_space=ac_space, softmax_actor=softmax_actor, nn_size=nn_size, init_scale=init_scale, advantage_learning=advantage_learning, use_layer_norm=use_layer_norm,
                         use_batch_norm=use_batch_norm, use_norm_actor=use_norm_actor, log_transform_inputs=log_transform_inputs)
        with tf.variable_scope(name):
            self._setup_states_feed()
            self._setup_actor()
            self._setup_critic()
            self._setup_training(
                a_lr=a_lr, a_l2_reg=a_l2_reg, a_clip_norm=a_clip_norm, lr=lr, l2_reg=l2_reg, clip_norm=clip_norm)
            self._setup_saving_loading_ops()

    def _setup_actor_training(self, a_l2_reg, a_clip_norm):
        with tf.variable_scope('optimize_actor'):
            # for training actions: maximize Advantage i.e. A
            self._a_vars = self._get_tf_trainable_variables('actor')
            self._av_A = tf.reduce_mean(self._A)
            loss = -self._av_A
            if a_l2_reg > 0:
                with tf.name_scope('L2_Losses'):
                    l2_loss = 0
                    for var in self._a_vars:
                        if 'bias' not in var.name and 'output' not in var.name:
                            l2_loss += a_l2_reg * tf.nn.l2_loss(var)
                loss = loss + l2_loss
            update_ops = self._get_update_ops('actor')
            with tf.control_dependencies(update_ops):
                a_grads = tf.gradients(loss, self._a_vars)
                if a_clip_norm is not None:
                    a_grads = [tf.clip_by_norm(
                        grad, clip_norm=a_clip_norm) for grad in a_grads]
                self._train_a_op = self._optimizer_a.apply_gradients(
                    list(zip(a_grads, self._a_vars)))
                # self.train_a_op = optimizer_a.minimize(-self.av_A, var_list=self.a_vars)

            # for supervised training of actions:
            self._a_desired = tf.placeholder(dtype=tf.float32, shape=[
                                             None] + list(self.ac_shape), name='desired_actions_feed')
            se = tf.square(self._a - self._a_desired)
            self._a_mse = tf.reduce_mean(se)
            loss = self._a_mse
            if a_l2_reg > 0:
                loss = loss + l2_loss
            with tf.control_dependencies(update_ops):
                a_grads = tf.gradients(loss, self._a_vars)
                if a_clip_norm is not None:
                    a_grads = [tf.clip_by_norm(
                        grad, clip_norm=a_clip_norm) for grad in a_grads]
                self._train_a_supervised_op = self._optimizer_a_supervised.apply_gradients(
                    list(zip(a_grads, self._a_vars)))

    def _setup_critic_training(self, l2_reg, clip_norm):
        if self.advantage_learning:
            with tf.variable_scope('optimize_V'):
                # for training V:
                self._V_vars = self._get_tf_trainable_variables('critic/V')
                self._V_target_feed = tf.placeholder(
                    dtype='float32', shape=[None])
                se = tf.square(self._V - self._V_target_feed)
                self._V_mse = tf.reduce_mean(se)
                loss = self._V_mse
                if l2_reg > 0:
                    with tf.variable_scope('L2_Losses'):
                        l2_loss = 0
                        for var in self._V_vars:
                            if 'bias' not in var.name and 'output' not in var.name:
                                l2_loss += l2_reg * tf.nn.l2_loss(var)
                        loss = loss + l2_loss
                update_ops = self._get_update_ops('critic/V')
                with tf.control_dependencies(update_ops):
                    V_grads = tf.gradients(loss, self._V_vars)
                    if clip_norm is not None:
                        V_grads = [tf.clip_by_norm(
                            grad, clip_norm=clip_norm) for grad in V_grads]
                    self._train_V_op = self._optimizer_V.apply_gradients(
                        list(zip(V_grads, self._V_vars)))
                    # self.train_V_op = optimizer_q.minimize(self.V_mse, var_list=self.V_vars)

        with tf.variable_scope('optimize_A'):
            # for training A:
            self._A_vars = self._get_tf_trainable_variables('critic/A')
            self._A_target_feed = tf.placeholder(dtype='float32', shape=[None])
            se = tf.square(self._A - self._A_target_feed)
            self._A_mse = tf.reduce_mean(se)
            loss = self._A_mse
            if l2_reg > 0:
                with tf.variable_scope('L2_Losses'):
                    l2_loss = 0
                    for var in self._A_vars:
                        if 'bias' not in var.name and 'output' not in var.name:
                            l2_loss += l2_reg * tf.nn.l2_loss(var)
                loss = loss + l2_loss
            update_ops = self._get_update_ops('critic/A')
            with tf.control_dependencies(update_ops):
                A_grads = tf.gradients(loss, self._A_vars)
                if clip_norm is not None:
                    A_grads = [tf.clip_by_norm(
                        grad, clip_norm=clip_norm) for grad in A_grads]
                self._train_A_op = self._optimizer_A.apply_gradients(
                    list(zip(A_grads, self._A_vars)))
                # self.train_A_op = optimizer_q.minimize(self.A_mse, var_list=self.A_vars)

    def _setup_training(self, a_lr, a_l2_reg, a_clip_norm, lr, l2_reg, clip_norm):
        with tf.variable_scope('training'):
            with tf.variable_scope('optimizers'):
                self._optimizer_A = tf.train.AdamOptimizer(
                    learning_rate=lr, name='A_adam')
                self._optimizer_V = tf.train.AdamOptimizer(
                    learning_rate=lr, name='V_adam')
                self._optimizer_a = tf.train.AdamOptimizer(
                    learning_rate=a_lr, name='actor_adam')
                self._optimizer_a_supervised = tf.train.AdamOptimizer(
                    learning_rate=a_lr, name='actor_supervised_adam')
            self._setup_actor_training(a_l2_reg, a_clip_norm)
            self._setup_critic_training(l2_reg, clip_norm)

    def _setup_saving_loading_ops(self):
        with tf.variable_scope('saving_loading_ops'):
            # for saving and loading
            params = self._get_tf_variables()
            self._load_placeholders = []
            self._load_ops = []
            for p in params:
                p_placeholder = tf.placeholder(
                    shape=p.shape.as_list(), dtype=tf.float32)
                self._load_placeholders.append(p_placeholder)
                self._load_ops.append(p.assign(p_placeholder))

    def train_V(self, states, target_V):
        return self.session.run([self._train_V_op, self._V_mse], feed_dict={
            self._states_feed: states,
            self._V_target_feed: target_V,
            self._is_training_critic: True
        })

    def train_A(self, states, target_A, actions=None):
        use_actions_feed = actions is not None
        if actions is None:
            actions = self.DUMMY_ACTION
        return self.session.run([self._train_A_op, self._A_mse], feed_dict={
            self._states_feed: states,
            self._use_actions_feed: use_actions_feed,
            self._actions_feed: actions,
            self._A_target_feed: target_A,
            self._is_training_a: False,
            self._is_training_critic: True
        })

    def train_Q(self, states, target_Q, actions=None):
        return self.train_A(states, target_Q, actions=actions)

    def train_a(self, states):
        return self.session.run([self._train_a_op, self._av_A], feed_dict={
            self._states_feed: states,
            self._use_actions_feed: False,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: True,
            self._is_training_critic: False
        })

    def train_a_supervised(self, states, desired_actions):
        return self.session.run([self._train_a_supervised_op, self._a_mse], feed_dict={
            self._states_feed: states,
            self._a_desired: desired_actions,
            self._use_actions_feed: False,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: True,
            self._is_training_critic: False
        })

    def save(self, save_path):
        params = self.session.run(self._get_tf_variables())
        from baselines.a2c.utils import make_path
        make_path(os.path.dirname(save_path))
        joblib.dump(params, save_path)

    def load(self, load_path):
        params = joblib.load(load_path)
        feed_dict = {}
        for p, p_placeholder in zip(params, self._load_placeholders):
            feed_dict[p_placeholder] = p
        self.session.run(self._load_ops, feed_dict=feed_dict)


class DDPG_Model_Target(DDPG_Model_Base):
    def __init__(self, session: tf.Session, name, main_network: DDPG_Model_Main, ob_space, ac_space, softmax_actor, nn_size, init_scale, advantage_learning, use_layer_norm, use_batch_norm, use_norm_actor, tau, log_transform_inputs, **kwargs):
        super().__init__(session=session, name=name, ob_space=ob_space, ac_space=ac_space, softmax_actor=softmax_actor, nn_size=nn_size, init_scale=init_scale, advantage_learning=advantage_learning,
                         use_layer_norm=use_layer_norm, use_batch_norm=use_batch_norm, use_norm_actor=use_norm_actor, log_transform_inputs=log_transform_inputs)
        self.main_network = main_network
        self.tau = tau
        with tf.variable_scope(name):
            self._setup_states_feed()
            self._setup_actor()
            self._setup_critic()
            self._setup_update_from_main_network()

    def _setup_update_from_main_network(self):
        with tf.variable_scope('target_network_update_ops'):
            # for updating target network:
            from_vars = self.main_network._get_tf_variables()
            from_vars_trainable = self.main_network._get_tf_trainable_variables()
            to_vars = self._get_tf_variables()
            to_vars_trainable = self.main_network._get_tf_trainable_variables()
            assert len(from_vars) == len(to_vars) and len(from_vars) > 0, print(
                '{0},{1}'.format(len(from_vars), len(to_vars)))
            assert len(from_vars_trainable) == len(to_vars_trainable) and len(from_vars_trainable) > 0, print(
                '{0},{1}'.format(len(from_vars_trainable), len(to_vars_trainable)))
            self._update_network_op, self._soft_update_network_op = [], []
            for from_var, to_var in zip(from_vars, to_vars):
                hard_update_op = to_var.assign(from_var)
                soft_update_op = to_var.assign(
                    self.tau * from_var + (1 - self.tau) * to_var)
                self._update_network_op.append(hard_update_op)
                # if from_var not in from_vars_trainable:
                #     soft_update_op = hard_update_op
                self._soft_update_network_op.append(soft_update_op)

    def soft_update_from_main_network(self):
        self.session.run(self._soft_update_network_op)

    def update_from_main_network(self):
        self.session.run(self._update_network_op)


class DDPG_Model_With_Param_Noise(DDPG_Model_Base):
    def __init__(self, session: tf.Session, name, main_network: DDPG_Model_Main, target_divergence, ob_space, ac_space, softmax_actor, nn_size, init_scale, advantage_learning, use_layer_norm, use_batch_norm, use_norm_actor, log_transform_inputs, **kwargs):
        super().__init__(session=session, name=name, ob_space=ob_space, ac_space=ac_space, softmax_actor=softmax_actor, nn_size=nn_size, init_scale=init_scale, advantage_learning=advantage_learning,
                         use_layer_norm=use_layer_norm, use_batch_norm=use_batch_norm, use_norm_actor=use_norm_actor, log_transform_inputs=log_transform_inputs)
        self.main_network = main_network
        self.target_divergence = target_divergence
        self._main_network_actor_params = self.main_network._get_tf_variables(
            'actor')
        self._main_network_actor_params_perturbable = self.main_network._get_tf_perturbable_variables(
            'actor')
        with tf.variable_scope(name):
            self._setup_states_feed()
            self._setup_actor()
            # dont setup critic for this one
            self._setup_update_from_main_network()
            self._setup_divergence_calculation()
            self._setup_param_sensitivity_calculation()
        self.adaptive_sigma = init_scale

    def _setup_update_from_main_network(self):
        with tf.variable_scope('noisy_actor_update_ops'):
            from_vars = self._main_network_actor_params
            to_vars = self._get_tf_variables()
            to_vars_perturbable = self._get_tf_perturbable_variables()
            assert len(from_vars) == len(to_vars) and len(from_vars) > 0, print(
                '{0},{1}'.format(len(from_vars), len(to_vars)))
            assert len(self._main_network_actor_params_perturbable) == len(to_vars_perturbable) and len(to_vars_perturbable) > 0, print(
                '{0},{1}'.format(len(self._main_network_actor_params_perturbable), len(to_vars_perturbable)))
            self._noise_vars = []
            self._noisy_update_network_op = []
            for from_var, to_var in zip(from_vars, to_vars):
                if from_var in self._main_network_actor_params_perturbable:
                    noise_var = tf.placeholder(
                        shape=from_var.shape.as_list(), dtype=tf.float32)
                    self._noise_vars.append(noise_var)
                    self._noisy_update_network_op.append(
                        to_var.assign(from_var + noise_var))
                else:
                    self._noisy_update_network_op.append(
                        to_var.assign(from_var))

    def _setup_param_sensitivity_calculation(self):
        with tf.variable_scope('actor_params_sensitivities'):
            sensitivities_squared = [
                0] * len(self._main_network_actor_params_perturbable)
            for k in range(self.ac_shape[0]):
                gradients_k = tf.gradients(
                    self.main_network._a[:, k], self._main_network_actor_params_perturbable)
                for var_index in range(len(self._main_network_actor_params_perturbable)):
                    sensitivities_squared[var_index] = sensitivities_squared[var_index] + tf.square(
                        gradients_k[var_index])
            self._main_network_actor_params_sensitivities = [
                tf.sqrt(s) for s in sensitivities_squared]

    def _setup_divergence_calculation(self):
        with tf.variable_scope('divergence'):
            self._divergence = tf.sqrt(tf.reduce_mean(
                tf.square(self._a - self.main_network._a)))

    def noisy_update_from_main_network(self, param_noise):
        feed_dict = {}
        for noise_var_placeholder, noise_var in zip(self._noise_vars, param_noise):
            feed_dict[noise_var_placeholder] = noise_var
        return self.session.run(self._noisy_update_network_op, feed_dict=feed_dict)

    def get_divergence(self, states):
        return self.session.run(self._divergence, feed_dict={
            self._states_feed: states,
            self._use_actions_feed: False,
            self._actions_feed: self.DUMMY_ACTION,
            self._is_training_a: False,
            self.main_network._states_feed: states,
            self.main_network._use_actions_feed: False,
            self.main_network._actions_feed: self.DUMMY_ACTION,
            self.main_network._is_training_a: False
        })

    def generate_normal_param_noise(self, sigma=None):
        if sigma is None:
            sigma = self.adaptive_sigma
        params = self.session.run(self._main_network_actor_params_perturbable)
        noise = []
        for p in params:
            n = sigma * np.random.standard_normal(size=np.shape(p))
            noise.append(n)
        return noise

    def generate_safe_noise(self, states, sigma=None):
        noise = self.generate_normal_param_noise(sigma)
        feed_dict = {
            self.main_network._states_feed: states,
            self.main_network._is_training_a: False,
            self.main_network._use_actions_feed: False,
            self.main_network._actions_feed: self.DUMMY_ACTION
        }
        sensitivities = self.session.run(
            self._main_network_actor_params_sensitivities, feed_dict=feed_dict)
        noise_safe = []
        for n, s in zip(noise, sensitivities):
            s = s / np.sqrt(len(states))  # to make s independent of mb size
            n_safe = n / (s + 1e-3)
            noise_safe.append(n_safe)
        return noise_safe

    def adapt_sigma(self, divergence):
        multiplier = 1 + abs(divergence - self.target_divergence)
        if divergence < self.target_divergence:
            self.adaptive_sigma = self.adaptive_sigma * multiplier
        else:
            self.adaptive_sigma = self.adaptive_sigma / multiplier


class Summaries:
    def __init__(self, session: tf.Session):
        self.session = session
        self.writer = tf.summary.FileWriter(
            logger.get_dir(), self.session.graph)

    def setup_scalar_summaries(self, keys):
        for k in keys:
            # ensure no white spaces in k:
            if ' ' in k:
                raise ValueError("Keys cannot contain whitespaces")
            placeholder_symbol = k
            setattr(self, placeholder_symbol, tf.placeholder(
                dtype=tf.float32, name=placeholder_symbol + '_placeholder'))
            placeholder = getattr(self, placeholder_symbol)
            summay_symbol = k + '_summary'
            setattr(self, summay_symbol, tf.summary.scalar(k, placeholder))

    def setup_histogram_summaries(self, keys):
        for k in keys:
            # ensure no white spaces in k:
            if ' ' in k:
                raise ValueError("Keys cannot contain whitespaces")
            placeholder_symbol = k
            setattr(self, placeholder_symbol, tf.placeholder(
                dtype=tf.float32, shape=[None], name=placeholder_symbol + '_placeholder'))
            placeholder = getattr(self, placeholder_symbol)
            summay_symbol = k + '_summary'
            setattr(self, summay_symbol, tf.summary.histogram(k, placeholder))

    def write_summaries(self, kvs, global_step):
        for key in kvs:
            placeholder_symbol = key
            summary_symbol = key + "_summary"
            if hasattr(self, placeholder_symbol) and hasattr(self, summary_symbol):
                summary = self.session.run(getattr(self, summary_symbol), feed_dict={
                    getattr(self, placeholder_symbol): kvs[key]
                })
                self.writer.add_summary(summary, global_step=global_step)
            else:
                logger.log("Invalid summary key {0}".format(
                    key), level=logger.WARN)


class DDPG_Model:
    def __init__(self, session: tf.Session, use_param_noise, sys_args_dict):
        self.main = DDPG_Model_Main(session, "model_main", **sys_args_dict)
        self.target = DDPG_Model_Target(
            session, "model_target", self.main, **sys_args_dict)
        if use_param_noise:
            print("creating noisy actor")
            self.noisy = DDPG_Model_With_Param_Noise(
                session, "model_param_noise", self.main, **sys_args_dict)
        self.summaries = Summaries(session)
