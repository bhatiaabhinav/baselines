"""
dualing network style advantageous DDPG
"""
from collections import deque
import os
import os.path
import sys
from typing import List  # noqa: F401

import gym
import joblib
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import gym_ERSLE  # noqa: F401
from baselines import logger
from baselines.a2c.utils import conv, conv_to_fc, fc
from baselines.common.atari_wrappers import FrameStack, EpisodicLifeEnv,\
    NoopResetEnv, MaxEnv, FireResetEnv, WarpFrame, SkipAndFrameStack,\
    ClipRewardEnv, BreakoutContinuousActionWrapper


class Actor:

    def __init__(self, session: tf.Session, name, ob_shape, ac_shape, ob_dtype='float32', q_lr=1e-3, a_lr=1e-4, use_layer_norm=True, tau=0.001):
        assert len(ac_shape) == 1
        self.session = session
        self.name = name
        self.ac_shape = ac_shape
        self.tau = tau

        def conv_layers(inputs, inputs_shape, scope):
            with tf.variable_scope(scope):
                if inputs_shape[0] >= 60:
                    c1 = conv(inputs, 'c1', nf=32, rf=8,
                              stride=4, init_scale=np.sqrt(2))
                c2 = conv(c1 if inputs_shape[0] >= 60 else inputs,
                          'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))
                c3 = conv(c2, 'c3', nf=64, rf=3,
                          stride=1, init_scale=np.sqrt(2))
                flat = conv_to_fc(c3)
            return flat

        def hidden_layers(inputs, scope, size, training=False):
            with tf.variable_scope(scope):
                for i in range(len(size)):
                    with tf.variable_scope('h{0}'.format(i + 1)):
                        h = fc(inputs, 'fc', size[i], act=lambda x: x)
                        if use_layer_norm:
                            h = tc.layers.layer_norm(
                                h, center=True, scale=True)
                        elif use_batch_norm:
                            h = tf.layers.batch_normalization(
                                h, training=training, name='batch_norm')
                        h = tf.nn.relu(h, name='relu')
                        inputs = h
            return inputs

        def deep_net(inputs, inputs_shape, inputs_dtype, scope, hidden_size, training=False, output_size=None):
            conv_needed = len(inputs_shape) > 1
            with tf.variable_scope(scope):
                if conv_needed:
                    inp = tf.cast(inputs, tf.float32, name='cast_to_float') / 255. \
                        if inputs_dtype == 'uint8' else inputs
                    flat = conv_layers(inp, inputs_shape, 'conv_layers')
                else:
                    flat = inputs
                h = hidden_layers(flat, 'hidden_layers',
                                  hidden_size, training=training)
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
                                     nn_size, training=is_training_a, output_size=ac_shape[0])
                        a = safe_softmax(
                            a, 'softmax') if 'ERS' in env_id else tf.nn.tanh(a, 'tanh')
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
                                         nn_size[0:1], training=is_training_critic)
                            s_a_concat = tf.concat(
                                [s, a], axis=-1, name="s_a_concat")
                            A = deep_net(s_a_concat, [nn_size[0] + ac_shape[0]], 'float32', 'A_network',
                                         nn_size[1:], training=is_training_critic, output_size=1)[:, 0]
                        V = deep_net(states_feed, ob_shape, ob_dtype, 'V', nn_size,
                                     training=is_training_critic, output_size=1)[:, 0]
                        # Q:
                        Q = tf.add(V, A, name='Q')
                        if scope == 'target':
                            self.V_target, self.A_target, self.Q_target = V, A, Q
                        else:
                            self.V, self.A, self.Q = V, A, Q
                            self.is_training_critic = is_training_critic

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
        self.score_summary = tf.summary.merge(
            [self.R_summary, self.R_exploit_summary])

    def get_a_V_A_Q(self, states):
        ops, feed = self.get_a_V_A_Q_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)

    def get_a_V_A_Q_op_and_feed(self, states):
        return [self.a, self.V, self.A, self.Q, self.frame_A_summary, self.frame_Q_summary],\
            {self.states_feed: states, self.use_actions_feed: False,
                self.actions_feed: [np.zeros(self.ac_shape)],
                self.is_training_a: False, self.is_training_critic: False}

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


class Experience:
    def __init__(self, state, action, reward, done, info, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
        self.next_state = next_state

    def __sizeof__(self):
        return sys.getsizeof(self.state) + sys.getsizeof(self.action) + \
            sys.getsizeof(self.reward) + sys.getsizeof(self.done) + \
            sys.getsizeof(self.info) + sys.getsizeof(self.next_state)


class ExperienceBuffer:
    def __init__(self, length=1e6, size_in_bytes=None):
        self.buffer = []  # type: List[Experience]
        self.buffer_length = length
        self.count = 0
        self.size_in_bytes = size_in_bytes
        self.next_index = 0

    def __len__(self):
        return self.count

    def add(self, exp: Experience):
        if self.count == 0:
            if self.size_in_bytes is not None:
                self.buffer_length = int(
                    self.size_in_bytes / sys.getsizeof(exp))
            logger.log('Initializing experience buffer of length {0}'.format(
                self.buffer_length))
            self.buffer = [None] * self.buffer_length
        self.buffer[self.next_index] = exp
        self.next_index = (self.next_index + 1) % self.buffer_length
        self.count = min(self.count + 1, self.buffer_length)

    def random_experiences(self, count):
        indices = np.random.randint(0, self.count, size=count)
        # yield self.buffer[(self.next_index - 1) % self.buffer_length]
        for i in indices:
            yield self.buffer[i]


class OrnsteinUhlenbeckActionNoise:
    '''Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab'''

    def __init__(self, mu, sigma=0.2, theta=6, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NormalNoise:
    def __init__(self, mu, sigma=0.2):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return self.mu + self.sigma * np.random.standard_normal(size=self.mu.shape)

    def reset(self):
        pass


class CartPoleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1., 1., shape=[1])

    def step(self, action):
        if action[0] < 0:
            a = 0
        else:
            a = 1
        return super().step(a)


class DiscreteToContinousWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(0, 1, shape=[env.action_space.n])

    def step(self, action):
        a = np.argmax(action)
        return super().step(a)


class ERSEnvWrapper(gym.Wrapper):
    k = 3

    def __init__(self, env):
        super().__init__(env)
        self.k = ERSEnvWrapper.k
        self.request_heat_maps = deque([], maxlen=self.k)
        self.n_ambs = 24
        self.n_bases = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(0, 1, shape=[self.n_bases])
        self.observation_space = gym.spaces.Box(
            0, 1, shape=[self.k * self.n_bases + self.n_bases + 1])

    def compute_alloc(self, action):
        action = np.clip(action, 0, 1)
        remaining = 1
        alloc = np.zeros([self.n_bases])
        for i in range(len(action)):
            alloc[i] = action[i] * remaining
            remaining -= alloc[i]
        alloc[-1] = remaining
        assert all(alloc >= 0) and all(
            alloc <= 1), "alloc is {0}".format(alloc)
        # assert sum(alloc) == 1, "sum is {0}".format(sum(alloc))
        return alloc

    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs = self.env.reset()
        for _ in range(self.k):
            self.request_heat_maps.append(self.obs[0:self.n_bases])
        return self._observation()

    def step(self, action):
        # action = self.compute_alloc(action)
        logger.log('alloc: {0}'.format(
            np.round(action * self.n_ambs, 2)), level=logger.DEBUG)
        self.obs, r, d, _ = super().step(action)
        self.request_heat_maps.append(self.obs[0:self.n_bases])
        return self._observation(), r, d, _

    def _observation(self):
        assert len(self.request_heat_maps) == self.k
        obs = np.concatenate((np.concatenate(
            self.request_heat_maps, axis=0), self.obs[self.n_bases:]), axis=0)
        logger.log('req_heat_map: {0}'.format(
            np.round(self.obs[0:self.n_bases], 2)), level=logger.DEBUG)
        return obs


def normalize(a):
    a = np.clip(a, 0, 1)
    a = a + 1e-6
    a = a / np.sum(a)
    return a


def test_actor_on_env(sess, learning=False, actor=None, save_path=None, load_path=None):
    np.random.seed(seed)
    env = gym.make(env_id)  # type: gym.Env
    for W in wrappers:
        env = W(env)  # type: gym.Wrapper
    if actor is None:
        actor = Actor(sess, 'actor', env.observation_space.shape, env.action_space.shape,
                      ob_dtype=ob_dtype, q_lr=q_lr, a_lr=a_lr, use_layer_norm=use_layer_norm, tau=tau)
        sess.run(tf.global_variables_initializer())
    if load_path:
        try:
            actor.load(load_path)
            logger.log('model loaded')
        except Exception as ex:
            logger.log('Failed to load model. Reason = {0}'.format(
                ex), level=logger.ERROR)
    actor.update_target_networks()
    if learning:
        experience_buffer = ExperienceBuffer(
            size_in_bytes=replay_memory_size_in_bytes)
        noise = Noise_type(mu=np.zeros(
            env.action_space.shape), sigma=exploration_sigma, theta=exploration_theta)

    def Q(s, a):
        return actor.get_V_A_Q(s, a)[2]

    def max_Q(s):
        return actor.get_a_V_A_Q(s)[3]

    def V(s):
        return actor.get_a_V_A_Q(s)[1]

    def A(s, a):
        return actor.get_V_A_Q(s, a)[1]

    def max_A(s):
        return actor.get_a_V_A_Q(s)[2]

    def argmax_Q(s):
        return actor.get_a_V_A_Q(s)[0]

    def _Q(s, a):
        return actor.get_target_V_A_Q(s, a)[2]

    def _max_Q(s):
        return actor.get_target_a_V_A_Q(s)[3]

    def _A(s, a):
        return actor.get_target_V_A_Q(s, a)[1]

    def _max_A(s):
        return actor.get_target_a_V_A_Q(s)[2]

    def _V(s):
        return actor.get_target_a_V_A_Q(s)[1]

    def _argmax_Q(s):
        return actor.get_target_a_V_A_Q(s)[0]

    def train(pre_train=False):
        count = max(pre_training_steps if pre_train else 1, 1)
        for c in range(count):
            mb = list(experience_buffer.random_experiences(
                count=minibatch_size))  # type: List[Experience]
            s, a, s_next, r, d = [e.state for e in mb], [e.action for e in mb], [
                e.next_state for e in mb], np.asarray([e.reward for e in mb]), np.asarray([int(e.done) for e in mb])
            ɣ = (1 - d) * gamma

            if not double_Q_learning:
                # Single Q learning:
                if advantage_learning:
                    # A(s,a) <- r + ɣ * max[_Q(s_next, _)] - max[_Q(s, _)]
                    # V(s) <- max[_Q(s, _)]
                    old_max_Q_s = _max_Q(s)
                    adv_s_a = r + ɣ * _max_Q(s_next) - old_max_Q_s
                    v_s = old_max_Q_s
                else:
                    # Q(s,a) <- r + ɣ * max[_Q(s_next, _)]
                    # A function to act like Q now
                    adv_s_a = r + ɣ * _max_Q(s_next)
                    v_s = 0  # set V to zero
            else:
                if advantage_learning:
                    # Double Q learning:
                    # A(s,a) <- r + ɣ * _Q(s_next, argmax[Q(s_next, _)]) - _Q(s, argmax(s, _))
                    # V(s) <- _Q(s, argmax[Q(s, _)])
                    old_Q_s1_argmax_Q_s1 = _Q(s_next, argmax_Q(s_next))
                    old_Q_s_argmax_Q_s = _Q(s, argmax_Q(s))
                    adv_s_a = r + ɣ * old_Q_s1_argmax_Q_s1 - old_Q_s_argmax_Q_s
                    v_s = old_Q_s_argmax_Q_s
                else:
                    # Q <- r + ɣ * _Q(s_next, argmax[Q(s_next, _)])
                    # A function to act like Q now
                    adv_s_a = r + ɣ * _Q(s_next, argmax_Q(s_next))
                    v_s = 0  # set V to zero

            _, A_mse, A_mse_summary, A_summary = actor.train_A(
                s, adv_s_a, actions=a)
            if advantage_learning:
                _, V_mse, V_mse_summary, V_summary = actor.train_V(s, v_s)
            else:
                V_mse = 0

        _, av_max_A, av_max_A_summary = actor.train_a(s)

        if hard_update_target:
            if f % int(train_every / tau) == 0:  # every train_every/tau steps
                actor.update_target_networks()
        else:
            actor.soft_update_target_networks()

        if f % 100 == 0:
            logger.log('mb_V_mse: {0}\tmb_Av_V: {1}\tmb_A_mse: {2}\tmb_Av_A: {3}\tmb_Av_max_A: {4}'.format(
                V_mse, np.average(v_s), A_mse, np.average(adv_s_a), av_max_A))
            actor.writer.add_summary(A_mse_summary, f)
            actor.writer.add_summary(A_summary, f)
            if advantage_learning:
                actor.writer.add_summary(V_mse_summary, f)
                actor.writer.add_summary(V_summary, f)
            actor.writer.add_summary(av_max_A_summary, f)

    def act(obs):
        a, value, adv, q, A_summary, Q_summary = actor.get_a_V_A_Q([obs])
        a, value, adv, q = a[0], value[0], adv[0], q[0]
        if not no_explore:
            a += noise()
            if 'ERS' in env_id:
                a = normalize(a)
            else:
                a = env.action_space.high * np.clip(a, -1, 1)
        logger.log('ep_f: {0}\tA: {1}\tQ: {2}'.format(
            ep_l, adv, q), level=logger.DEBUG)
        if f % 100 == 0 and logger.Logger.CURRENT.level <= logger.DEBUG:
            actor.writer.add_summary(A_summary, f)
            actor.writer.add_summary(Q_summary, f)
        return a
    Rs, no_explore_Rs, f = [], [], 0
    env.seed(learning_env_seed if learning else test_env_seed)
    pre_train = True
    for ep in range(learning_episodes if learning else test_episodes):
        obs, d, R, ep_l = env.reset(), False, 0, 0
        if learning:
            noise.reset()
        no_explore = (ep % 2 == 0) or not learning
        while not d:
            if learning and ep >= exploration_episodes and f % train_every == 0:
                train(pre_train=pre_train)
                pre_train = False
            a = act(obs)
            obs_, r, d, _ = env.step(a)
            if render:
                env.render(mode=render_mode)
            if learning:
                experience_buffer.add(Experience(obs, a, r, d, _, obs_))
            obs, R, f, ep_l = obs_, R + r, f + 1, ep_l + 1
        Rs.append(R)
        if no_explore:
            no_explore_Rs.append(R)
        av = np.average(Rs[-100:])
        no_explore_av = np.average(no_explore_Rs[-100:])
        logger.log('Episode {0}:\tReward: {1}\tLength: {2}\tAv_R: {3}\tExploit_Av_R: {4}{5}'.format(
            ep, R, ep_l, av, no_explore_av, "\t(exploited)" if no_explore else ""))
        score_summary = actor.score_summary.eval(
            feed_dict={actor.R_placeholder: Rs[-1], actor.R_exploit_placeholder: no_explore_Rs[-1]})
        actor.writer.add_summary(score_summary, ep)
        if save_path and ep % 50 == 0:
            actor.save(save_path)
            logger.log('model saved')
    env.close()
    logger.log('Average reward per episode: {0}'.format(np.average(Rs)))
    logger.log('Exploitation average reward per episode: {0}'.format(
        np.average(no_explore_Rs)))
    return actor


if __name__ == '__main__':
    # config = tf.ConfigProto(device_count={'GPU': 0})
    from baselines.ers.args import parse
    args = parse()
    env_id = args.env
    logger.log('env_id: ' + env_id)
    seed = args.seed
    logger.log('Seed: {0}'.format(seed))
    np.random.seed(seed)
    ob_dtype = args.ob_dtype
    wrappers = []
    minibatch_size = args.mb_size
    tau = args.tau
    train_every = args.train_every
    hard_update_target = args.hard_update_target
    gamma = args.gamma
    q_lr = args.a_lr
    a_lr = args.a_lr
    clip_norm = args.clip_norm
    a_clip_norm = args.a_clip_norm
    l2_reg = args.l2_reg
    a_l2_reg = args.a_l2_reg
    double_Q_learning = args.double_Q_learning
    advantage_learning = args.advantage_learning
    exploration_episodes = args.exploration_episodes
    pre_training_steps = args.pre_training_steps
    replay_memory_size_in_bytes = args.replay_memory_gigabytes * 2**30
    exploration_sigma = args.exploration_sigma
    exploration_theta = args.exploration_theta
    Noise_type = OrnsteinUhlenbeckActionNoise
    learning_env_seed = seed
    learning_episodes = args.training_episodes
    test_env_seed = args.test_seed
    test_episodes = args.test_episodes
    test_mode = args.test_mode
    use_layer_norm = args.use_layer_norm
    use_batch_norm = args.use_batch_norm
    assert not (
        use_layer_norm and use_batch_norm), "Cannot use both layer norm and batch norm"
    nn_size = args.nn_size
    init_scale = args.init_scale
    render = args.render
    render_mode = args.render_mode
    save_path = os.path.join(logger.get_dir(), "model")
    load_path = args.saved_model
    ERSEnvWrapper.k = args.nstack
    FrameStack.k = args.nstack
    if 'ERSEnv-ca' in env_id:
        wrappers = [ERSEnvWrapper]
    elif 'ERSEnv-im' in env_id:
        wrappers = [FrameStack]
    elif 'Pole' in env_id:
        wrappers = [CartPoleWrapper]
    elif 'NoFrameskip' in env_id:
        wrappers = [EpisodicLifeEnv, NoopResetEnv, MaxEnv, FireResetEnv, WarpFrame,
                    SkipAndFrameStack, ClipRewardEnv, BreakoutContinuousActionWrapper]
    with tf.Session() as sess:
        if not test_mode:
            logger.log('Training actor. seed={0}. learning_env_seed={1}'.format(
                seed, learning_env_seed))
            actor = test_actor_on_env(
                sess, True, save_path=save_path, load_path=load_path)
            actor.save(save_path)
        logger.log('Testing actor. test_env_seed={0}'.format(test_env_seed))
        if not test_mode:
            test_actor_on_env(sess, learning=False, actor=actor)
        else:
            assert load_path is not None, "Please provide a saved model"
            test_actor_on_env(sess, learning=False, load_path=load_path)
        logger.log('Testing done. Seeds were seed={0}. learning_env_seed={1}. test_env_seed={2}'.format(
            seed, learning_env_seed, test_env_seed))
        logger.log('-------------------------------------------------\n')
