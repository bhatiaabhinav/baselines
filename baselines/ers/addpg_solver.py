"""
dualing network style advantageous DDPG
"""

import os
import os.path
import sys
from typing import List

import gym
import joblib
import numpy as np
import tensorflow as tf

import gym_ERSLE
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
        with tf.variable_scope(name):
            for scope in ['original', 'target']:
                with tf.variable_scope(scope):
                    states_feed = tf.placeholder(dtype=ob_dtype, shape=[
                                                 None] + list(ob_shape))
                    if scope == 'target':
                        self.states_feed_target = states_feed
                    else:
                        self.states_feed = states_feed
                    with tf.variable_scope('a'):
                        # conv layers go here
                        if len(ob_shape) > 1:
                            inp = tf.cast(states_feed, tf.float32) / 255.
                            if ob_shape[0] >= 60:
                                a_c1 = conv(inp, 'a_c1',
                                            nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
                            a_c2 = conv(a_c1 if ob_shape[0] >= 60 else inp, 'a_c2',
                                        nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
                            a_c3 = conv(a_c2, 'a_c3', nf=64, rf=3,
                                        stride=1, init_scale=np.sqrt(2))
                            states_flat = conv_to_fc(a_c3)
                        else:
                            states_flat = states_feed
                        a_h1 = fc(states_flat, 'a_h1',
                                  nh=nn_size[0], act=lambda x: x)
                        if use_layer_norm:
                            a_h1 = tf.layers.batch_normalization(a_h1)
                        a_h1 = tf.nn.relu(a_h1)
                        a_h2 = fc(a_h1, 'a_h2', nh=nn_size[1], act=lambda x: x)
                        if use_layer_norm:
                            a_h2 = tf.layers.batch_normalization(a_h2)
                        a_h2 = tf.nn.relu(a_h2)
                        if 'ERS' in env_id:
                            a = fc(a_h2, 'a', nh=ac_shape[0],
                                   act=lambda x: x, init_scale=init_scale)
                            exp = tf.exp(
                                a - tf.reduce_max(a, axis=-1, keep_dims=True))
                            a = exp / \
                                tf.reduce_sum(exp, axis=-1, keep_dims=True)
                        else:
                            a = fc(
                                a_h2, 'a', nh=ac_shape[0], act=tf.nn.tanh, init_scale=init_scale)
                        use_actions_feed = tf.placeholder(dtype=tf.bool)
                        actions_feed = tf.placeholder(
                            dtype=tf.float32, shape=[None] + list(ac_shape))
                        a = tf.case([
                            (use_actions_feed, lambda: actions_feed)
                        ], default=lambda: a)
                        if scope == 'target':
                            self.a_target = a
                            self.actions_feed_target = actions_feed
                            self.use_actions_feed_target = use_actions_feed
                        else:
                            self.a = a
                            self.actions_feed = actions_feed
                            self.use_actions_feed = use_actions_feed
                    with tf.variable_scope('q'):
                        # conv layers go here
                        if len(ob_shape) > 1:
                            inp = tf.cast(states_feed, tf.float32) / 255.
                            if ob_shape[0] >= 60:
                                s_c1 = conv(inp, 's_c1',
                                            nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
                            s_c2 = conv(s_c1 if ob_shape[0] >= 60 else inp, 's_c2',
                                        nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
                            s_c3 = conv(s_c2, 's_c3', nf=64, rf=3,
                                        stride=1, init_scale=np.sqrt(2))
                            states_flat = conv_to_fc(s_c3)
                        else:
                            states_flat = states_feed
                        s_h1 = fc(states_flat, 's_h1',
                                  nh=nn_size[0], act=lambda x: x)
                        if use_layer_norm:
                            s_h1 = tf.layers.batch_normalization(s_h1)
                        s_h1 = tf.nn.relu(s_h1)
                        s = s_h1
                        # the advantage network:
                        s_a_concat = tf.concat([s, a], axis=-1)
                        A_h1 = fc(s_a_concat, 'A_h1', nh=nn_size[1])
                        A = fc(A_h1, 'A', 1, act=lambda x: x,
                               init_scale=init_scale)[:, 0]
                        # the value network:
                        V_h1 = fc(s, 'V_h1', nh=nn_size[1])
                        V = fc(V_h1, 'V', 1, act=lambda x: x,
                               init_scale=init_scale)[:, 0]
                        # q:
                        Q = V + A
                        if scope == 'target':
                            self.V_target, self.A_target, self.Q_target = V, A, Q
                        else:
                            self.V, self.A, self.Q = V, A, Q

        # optimizers:
        optimizer_q = tf.train.AdamOptimizer(learning_rate=q_lr)
        optimizer_a = tf.train.AdamOptimizer(learning_rate=a_lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # for training actions: maximize Advantage i.e. A
        self.a_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/a'.format(name))
        self.av_A = tf.reduce_mean(self.A)
        with tf.control_dependencies(update_ops):
            a_grads = tf.gradients(-self.av_A, self.a_vars)
            a_grads, norm = tf.clip_by_global_norm(a_grads, clip_norm=1)
            self.train_a_op = optimizer_a.apply_gradients(
                list(zip(a_grads, self.a_vars)))
            # self.train_a_op = optimizer_a.minimize(-self.av_A, var_list=self.a_vars)

        # for training V:
        self.V_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/q'.format(name))
        self.V_target_feed = tf.placeholder(dtype='float32', shape=[None])
        se = tf.square(self.V - self.V_target_feed) / 2
        self.V_mse = tf.reduce_mean(se)
        with tf.control_dependencies(update_ops):
            V_grads = tf.gradients(self.V_mse, self.V_vars)
            V_grads, norm = tf.clip_by_global_norm(V_grads, clip_norm=1)
            self.train_V_op = optimizer_q.apply_gradients(
                list(zip(V_grads, self.V_vars)))
            # self.train_V_op = optimizer_q.minimize(self.V_mse, var_list=self.V_vars)

        # for training A:
        self.A_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/q'.format(name))
        self.A_target_feed = tf.placeholder(dtype='float32', shape=[None])
        se = tf.square(self.A - self.A_target_feed) / 2
        self.A_mse = tf.reduce_mean(se)
        with tf.control_dependencies(update_ops):
            A_grads = tf.gradients(self.A_mse, self.A_vars)
            A_grads, norm = tf.clip_by_global_norm(A_grads, clip_norm=1)
            self.train_A_op = optimizer_q.apply_gradients(
                list(zip(A_grads, self.A_vars)))
            # self.train_A_op = optimizer_q.minimize(self.A_mse, var_list=self.A_vars)

        # for updating target network:
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      '{0}/original'.format(name))
        to_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/target'.format(name))
        self.update_target_network_op, self.soft_update_target_network_op = [], []
        for from_var, to_var in zip(from_vars, to_vars):
            self.update_target_network_op.append(to_var.assign(from_var))
            self.soft_update_target_network_op.append(
                to_var.assign(tau * from_var + (1 - tau) * to_var))

        # for saving and loading
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{0}'.format(name))
        self.load_placeholders = []
        self.load_ops = []
        for p in self.params:
            p_placeholder = tf.placeholder(
                shape=p.shape.as_list(), dtype=tf.float32)
            self.load_placeholders.append(p_placeholder)
            self.load_ops.append(p.assign(p_placeholder))

        # for visualizing computation graph in tensorboard
        self.writer = tf.summary.FileWriter(
            'summary/addpg', self.session.graph)

    def get_a_V_A_Q(self, states):
        ops, feed = self.get_a_V_A_Q_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)

    def get_a_V_A_Q_op_and_feed(self, states):
        return [self.a, self.V, self.A, self.Q], {self.states_feed: states, self.use_actions_feed: False, self.actions_feed: [np.zeros(self.ac_shape)]}

    def get_V_A_Q(self, states, actions):
        ops, feed = self.get_V_A_Q_op_and_feed(states, actions)
        return self.session.run(ops, feed_dict=feed)

    def get_V_A_Q_op_and_feed(self, states, actions):
        return [self.V, self.A, self.Q], {self.states_feed: states, self.use_actions_feed: True, self.actions_feed: actions}

    def train_V(self, states, target_V):
        ops, feed = self.get_train_V_op_and_feed(
            states, target_V)
        return self.session.run(ops, feed_dict=feed)

    def get_train_V_op_and_feed(self, states, target_V):
        return [self.train_V_op, self.V_mse], {self.states_feed: states, self.V_target_feed: target_V}

    def train_A(self, states, target_A, actions=None):
        ops, feed = self.get_train_A_op_and_feed(
            states, target_A, actions=actions)
        return self.session.run(ops, feed_dict=feed)

    def get_train_A_op_and_feed(self, states, target_A, actions=None):
        use_actions_feed = actions is not None
        if actions is None:
            actions = [np.zeros(self.ac_shape)]
        return [self.train_A_op, self.A_mse], {self.states_feed: states, self.use_actions_feed: use_actions_feed, self.actions_feed: actions, self.A_target_feed: target_A}

    def train_a(self, states):
        ops, feed = self.get_train_a_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)

    def get_train_a_op_and_feed(self, states):
        return [self.train_a_op, self.av_A], {self.states_feed: states, self.use_actions_feed: False, self.actions_feed: [np.zeros(self.ac_shape)]}

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
            print('Initializing experience buffer of length {0}'.format(
                self.buffer_length))
            self.buffer = [None] * self.buffer_length
        self.buffer[self.next_index] = exp
        self.next_index = (self.next_index + 1) % self.buffer_length
        self.count = min(self.count + 1, self.buffer_length)

    def random_experiences(self, count):
        indices = np.random.randint(0, self.count, size=count - 1)
        yield self.buffer[(self.next_index - 1) % self.buffer_length]
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
    def __init__(self, env):
        super().__init__(env)
        self.n_ambs = 24
        self.n_bases = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(0, 1, shape=[self.n_bases])

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

    def step(self, action):
        # action = self.compute_alloc(action)
        print(np.round(self.n_ambs * action, decimals=2))
        obs, r, d, _ = super().step(action)
        assert list(obs.shape) == [21, 21, 21]
        r = r / 200
        return obs, r, d, _


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
                      ob_dtype=ob_dtype, q_lr=1e-3, a_lr=1e-4, use_layer_norm=use_layer_norm, tau=tau)
        sess.run(tf.global_variables_initializer())
    actor.update_target_networks()
    if load_path:
        try:
            actor.load(load_path)
            print('model loaded')
        except Exception as ex:
            print('Failed to load model. Reason = {0}'.format(ex))
    if learning:
        experience_buffer = ExperienceBuffer(
            size_in_bytes=replay_memory_size_in_bytes)
        noise = Noise_type(mu=np.zeros(
            env.action_space.shape), sigma=exploration_sigma)

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

    def _max_A(s):
        return actor.get_target_a_V_A_Q(s)[2]

    def _V(s):
        return actor.get_target_a_V_A_Q(s)[1]

    def _argmax_Q(s):
        return actor.get_target_a_V_A_Q(s)[0]

    def train(pre_train=False):
        count = pre_training_steps if pre_train else 1
        for c in range(count):
            mb = list(experience_buffer.random_experiences(
                count=minibatch_size))  # type: List[Experience]
            s, a, s_next, r, d = [e.state for e in mb], [e.action for e in mb], [
                e.next_state for e in mb], np.asarray([e.reward for e in mb]), np.asarray([int(e.done) for e in mb])
            g = (1 - d) * gamma

            a_s_cur, v_s_cur, max_A_cur, max_Q_cur = actor.get_a_V_A_Q(s)
            _, old_v_s, old_max_A, __ = actor.get_target_a_V_A_Q(s)
            # nomrally: A(s,a) = r + gamma * max[_Q(s_next, _)] - _V(s)
            # double Q: A(s,a) = r + gamma * _Q(s_next, argmax[Q(s_next, _)]) - _V(s)
            adv_s_a = old_max_A + r + g * _V(s_next) - old_v_s
            _, A_mse = actor.train_A(s, adv_s_a, actions=a)

            # V(s) = max(_Q(s, _))
            # v_s_target = _max_Q(s)
            v_s = old_v_s + max_A_cur - old_max_A
            _, V_mse = actor.train_V(s, v_s)

            # normalize A:
            adv_s_a = A(s, a) - old_max_A
            _, _ = actor.train_A(s, adv_s_a, actions=a)

            actor.soft_update_target_networks()

        _, av_A = actor.train_a(s)

        if f % 100 == 0:
            print('V_mse: {0}\tAv_V: {1}\tA_mse: {2}\tAv_A: {3}'.format(
                V_mse, np.average(v_s_cur), A_mse, av_A))

    def act(obs):
        if no_explore:
            return actor.get_a_V_A_Q([obs])[0][0]
        else:
            a, value, adv, q = actor.get_a_V_A_Q([obs])
            a, value, adv, q = a[0], value[0], adv[0], q[0]
        a += noise()
        a = normalize(a) if 'ERS' in env_id else np.clip(a, -1, 1)
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
            if learning and ep >= exploration_episodes and f % 4 == 0:
                train(pre_train=pre_train)
                pre_train = False
            a = act(obs)
            obs_, r, d, _ = env.step(a)
            if render:
                env.render()
            if learning:
                experience_buffer.add(Experience(obs, a, r, d, _, obs_))
            obs, R, f, ep_l = obs_, R + r, f + 1, ep_l + 1
        if 'ERS' in env_id:
            R = 200 * R
        Rs.append(R)
        if no_explore:
            no_explore_Rs.append(R)
        av = np.average(Rs[-100:])
        no_explore_av = np.average(no_explore_Rs[-100:])
        print('Episode {0}:\tReward: {1}\tLength: {2}\tAv_R: {3}\tExploit_Av_R: {4}{5}'.format(
            ep, R, ep_l, av, no_explore_av, "\t(exploited)" if no_explore else ""))
        if save_path and ep % 50 == 0:
            actor.save(save_path)
            print('model saved')
        if learning and env_id == 'CartPole-v1' and av > 490:
            break
    env.close()
    print('Average reward per episode: {0}'.format(np.average(Rs)))
    print('Exploitation average reward per episode: {0}'.format(
        np.average(no_explore_Rs)))
    return actor


if __name__ == '__main__':
    # config = tf.ConfigProto(device_count={'GPU': 0})
    from baselines.ers.args import parse
    args = parse()
    env_id = args.env
    print('env_id: ' + env_id)
    seed = args.seed
    print('Seed: {0}'.format(seed))
    np.random.seed(seed)
    ob_dtype = args.ob_dtype
    minibatch_size = args.mb_size
    tau = args.tau
    gamma = args.gamma
    exploration_episodes = args.exploration_episodes
    pre_training_steps = args.pre_training_steps
    replay_memory_size_in_bytes = args.replay_memory_gigabytes * 2**30
    exploration_sigma = args.exploration_sigma
    Noise_type = OrnsteinUhlenbeckActionNoise
    learning_env_seed = seed
    learning_episodes = args.training_episodes
    test_env_seed = args.test_seed
    test_episodes = args.test_episodes
    use_layer_norm = args.use_layer_norm
    nn_size = args.nn_size
    init_scale = args.init_scale
    render = args.render
    save_path = 'models/{0}/{1}/model'.format(env_id, seed)
    load_path = save_path
    if 'ERSEnv-ca' in env_id:
        ob_dtype = args.ob_dtype
        wrappers = [ERSEnvWrapper]
    elif 'ERSEnv-im' in env_id:
        ob_dtype = 'uint8'
        wrappers = [FrameStack, ERSEnvWrapper]
    elif 'Pole' in env_id:
        ob_dtype = 'float32'
        wrappers = [CartPoleWrapper]
    elif 'NoFrameskip' in env_id:
        ob_dtype = 'uint8'
        wrappers = [EpisodicLifeEnv, NoopResetEnv, MaxEnv, FireResetEnv, WarpFrame,
                    SkipAndFrameStack, ClipRewardEnv, BreakoutContinuousActionWrapper]
    with tf.Session() as sess:
        print('Training actor. seed={0}. learning_env_seed={1}'.format(
            seed, learning_env_seed))
        actor = test_actor_on_env(
            sess, True, save_path=save_path, load_path=load_path)
        actor.save(save_path)
        print('Testing actor. test_env_seed={0}'.format(test_env_seed))
        test_actor_on_env(sess, False, actor=actor)
        print('Testing done. Seeds were seed={0}. learning_env_seed={1}. test_env_seed={2}'.format(
            seed, learning_env_seed, test_env_seed))
        print('-------------------------------------------------\n')
