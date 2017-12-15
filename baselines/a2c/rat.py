from typing import List
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc
import gym
import gym_ERSLE
import matplotlib.pyplot as plt
from baselines import logger
import os.path
import joblib
import logging
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import sys

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
                    states_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
                    if scope == 'target':
                        self.states_feed_target = states_feed
                    else:
                        self.states_feed = states_feed
                    with tf.variable_scope('a'):
                        # conv layers go here
                        states_flat = states_feed
                        a_h1 = fc(states_flat, 'a_h1', nh=nn_size[0], act=lambda x:x)
                        if use_layer_norm: a_h1 = tf.layers.batch_normalization(a_h1)
                        a_h1 = tf.nn.relu(a_h1)
                        a_h2 = fc(a_h1, 'a_h2', nh=nn_size[1], act=lambda x:x)
                        if use_layer_norm: a_h2 = tf.layers.batch_normalization(a_h2)
                        a_h2 = tf.nn.relu(a_h2)
                        if 'ERS' in env_id:
                            a = fc(a_h2, 'a', nh=ac_shape[0], act=lambda x:x, init_scale=1e-3)
                            exp = tf.exp(a - tf.reduce_max(a, axis=-1, keep_dims=True))
                            a = exp/tf.reduce_sum(exp, axis=-1, keep_dims=True)
                        else:
                            a = fc(a_h2, 'a', nh=ac_shape[0], act=tf.nn.tanh, init_scale=1e-3)
                        if scope == 'target':
                            self.a_target = a
                        else:
                            self.use_actions_feed = tf.placeholder(dtype=tf.bool)
                            self.actions_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ac_shape))
                            self.a = tf.case([
                                (self.use_actions_feed, lambda: self.actions_feed)
                            ], default=lambda: a)
                            a = self.a
                    with tf.variable_scope('q'):
                        # conv layers go here
                        states_flat = states_feed
                        s_h1 = fc(states_flat, 's_h1', nh=nn_size[0], act=lambda x:x)
                        if use_layer_norm: s_h1 = tf.layers.batch_normalization(s_h1)
                        s_h1 = tf.nn.relu(s_h1)
                        s = s_h1
                        s_a_concat = tf.concat([s, a], axis=-1)
                        # one hidden layer after concating s,a
                        q_h1 = fc(s_a_concat, 'q_h1', nh=nn_size[1])
                        q = fc(q_h1, 'q', 1, act=lambda x: x, init_scale=1e-3)[:, 0]
                        if scope == 'target':
                            self.q_target = q
                        else:
                            self.q = q
            
        # optimizers:
        optimizer_q = tf.train.AdamOptimizer(learning_rate=q_lr)
        optimizer_a = tf.train.AdamOptimizer(learning_rate=a_lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # for training actions:
        self.a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/a'.format(name))
        self.av_q = tf.reduce_mean(self.q)
        with tf.control_dependencies(update_ops):
            self.train_a_op = optimizer_a.minimize(-self.av_q, var_list=self.a_vars)

        # for training Q:
        self.q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/q'.format(name))
        self.q_target_feed = tf.placeholder(dtype='float32', shape=[None])
        se = tf.square(self.q - self.q_target_feed)/2
        self.mse = tf.reduce_mean(se)
        
        with tf.control_dependencies(update_ops):
            self.train_q_op = optimizer_q.minimize(self.mse, var_list=self.q_vars)

        # for updating target Q network:
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/original/q'.format(name))
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/target/q'.format(name))
        self.update_target_q_network_op, self.soft_update_target_q_network_op = [], []
        for from_var,to_var in zip(from_vars,to_vars):
            self.update_target_q_network_op.append(to_var.assign(from_var))
            self.soft_update_target_q_network_op.append(to_var.assign(tau * from_var + (1 - tau) * to_var))

        # for updating target a network:
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/original/a'.format(name))
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/target/a'.format(name))
        self.update_target_a_network_op, self.soft_update_target_a_network_op = [], []
        for from_var,to_var in zip(from_vars,to_vars):
            self.update_target_a_network_op.append(to_var.assign(from_var))
            self.soft_update_target_a_network_op.append(to_var.assign(tau * from_var + (1 - tau) * to_var))

        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}'.format(name))

    def get_actions_and_q(self, states):
        ops, feed = self.get_actions_and_q_op_and_feed_dict(states)
        return self.session.run(ops, feed_dict=feed)
    
    def get_actions_and_q_op_and_feed_dict(self, states):
        return [self.a, self.q], {self.states_feed:states, self.use_actions_feed: False, self.actions_feed: [np.zeros(self.ac_shape)]}
    
    def get_q(self, states, actions):
        ops, feed = self.get_q_op_and_feed(states, actions)
        return self.session.run(ops, feed_dict=feed)

    def get_q_op_and_feed(self, states, actions):
        return [self.q], {self.states_feed:states, self.use_actions_feed: True, self.actions_feed: actions}

    def train_q(self, states, target_q, actions=None):
        ops, feed = self.get_train_q_op_and_feed(states, target_q, actions=actions)
        return self.session.run(ops, feed_dict=feed)
    
    def get_train_q_op_and_feed(self, states, target_q, actions=None):
        if actions is None:
            actions = [np.zeros(self.ac_shape)]
        return [self.train_q_op, self.mse], {self.states_feed:states, self.use_actions_feed: (actions is not None), self.actions_feed: actions, self.q_target_feed: target_q}
    
    def train_a(self, states):
        ops, feed = self.get_train_a_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)
    
    def get_train_a_op_and_feed(self, states):
        return [self.train_a_op, self.av_q], {self.states_feed:states, self.use_actions_feed: False, self.actions_feed: [np.zeros(self.ac_shape)]}

    def get_target_actions_and_q(self, states):
        ops, feed = self.get_target_actions_and_q_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)

    def get_target_actions_and_q_op_and_feed(self, states):
        return [self.a_target, self.q_target], {self.states_feed_target: states}

    def update_target_networks(self):
        ops_q, feed = self.get_update_target_q_network_op_and_feed()
        ops_a, feed_a = self.get_update_target_a_network_op_and_feed()
        feed.update(feed_a)
        self.session.run(ops_q + ops_a, feed_dict=feed)

    def soft_update_target_networks(self):
        ops_q, feed = self.get_soft_update_target_q_network_op_and_feed()
        ops_a, feed_a = self.get_soft_update_target_a_network_op_and_feed()
        feed.update(feed_a)
        self.session.run(ops_q + ops_a, feed_dict=feed)

    def get_update_target_q_network_op_and_feed(self):
        return [self.update_target_q_network_op], {}

    def get_soft_update_target_q_network_op_and_feed(self):
        return [self.soft_update_target_q_network_op], {}

    def get_update_target_a_network_op_and_feed(self):
        return [self.update_target_a_network_op], {}

    def get_soft_update_target_a_network_op_and_feed(self):
        return [self.soft_update_target_a_network_op], {}

    def save(self, save_path):
        ps = self.session.run(self.params)
        from baselines.a2c.utils import make_path
        make_path(os.path.dirname(save_path))
        joblib.dump(ps, save_path)

    def load(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(self.params, loaded_params):
            restores.append(p.assign(loaded_p))
        ps = self.session.run(restores)

    def set_vars(self, newvars):
        feed_dict = {}
        for v, v_feed in zip(newvars, self.vars_feed_holder):
            feed_dict[v_feed] = v
        self.session.run(self.set_vars_op, feed_dict=feed_dict)

    def copy_to(self, other, mutation_probability=0, max_mutation=0.3):
        other = other # type: Recommender
        self_vars = self.get_vars()
        if mutation_probability > 0:
            for i in range(len(self_vars)):
                v = self_vars[i]
                mutations = np.random.standard_normal(np.shape(v)) * max_mutation
                mutations_gate = np.random.random_sample(np.shape(v)) < mutation_probability
                self_vars[i] = v + mutations_gate * mutations
        other.set_vars(self_vars)


# class EnsembledActors:
#     def __init__(self, session: tf.Session, , n_actors, ob_shape, ac_shape, ob_dtype='float32', q_lr=1e-3, a_lr=1e-4, use_layer_norm=True):
#         self.actors = 


# class ActorTrainer:

#     def __init__(self, session: tf.Session, recommenders: List[Recommender], ob_shape, ac_shape, ob_dtype='float32', learning_rate=1e-3):
#         # build the graph
#         self.recommenders = recommenders
#         self.session = session
#         self.ac_shape = ac_shape
#         self.ob_shape = ob_shape
#         with tf.variable_scope('actor_trainer'):
#             for scope in ['q', 'q_duplicate']:
#                 with tf.variable_scope(scope):
#                     if scope == 'q_duplicate':
#                         self.states_feed_duplicate = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
#                         self.recommenders_output = tf.concat([r.get_recommendations_tensor() for r in recommenders], axis=0)
#                         states = tf.concat([self.states_feed_duplicate]*len(recommenders), axis=0)
#                         actions = self.recommenders_output
#                     else:
#                         self.states_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
#                         self.actions_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ac_shape))
#                         states = self.states_feed
#                         actions = self.actions_feed
           
#                     # this will work only if state is single dimensional as well:
#                     inp = tf.concat((states, actions), axis=-1)
#                     h1 = fc(inp, 'fc1', nh=64, init_scale=np.sqrt(2))
#                     h2 = fc(h1, 'fc2', nh=32, init_scale=np.sqrt(2))
#                     h3 = fc(h2, 'fc3', nh=16, init_scale=np.sqrt(2))
#                     if scope == 'q_duplicate':
#                         self.q_duplicate = fc(h3, 'q', 1, act=lambda x: x)[:, 0]
#                     else:
#                         self.q = fc(h3, 'q', 1, act=lambda x: x)[:, 0]
        
#         optimizer_q = tf.train.AdamOptimizer(learning_rate=learning_rate)
#         optimizer_recommenders = tf.train.AdamOptimizer(learning_rate=learning_rate)

#         # for training the recommenders:
#         recommenders_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='recommenders')
#         self.av_q = tf.reduce_mean(self.q_duplicate)
#         self.train_recommenders_op = optimizer_recommenders.minimize(-self.av_q, var_list=recommenders_vars)

#         # for training the Q network:
#         q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_trainer/q')
#         self.target_q_feed = tf.placeholder(dtype='float32', shape=[None])
#         #clipped_diff = tf.clip_by_value(self.q - self.target_q_feed, -1, 1)
#         se = tf.square(self.q - self.target_q_feed)/2
#         self.mse = tf.reduce_mean(se)
#         self.train_q_op = optimizer_q.minimize(self.mse, var_list=q_vars)

#         # for updating duplicate q:
#         from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_trainer/q')
#         to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_trainer/q_duplicate')
#         op_holder = []
#         for from_var,to_var in zip(from_vars,to_vars):
#             op_holder.append(to_var.assign(from_var))
#         self.update_duplicate_q_op = op_holder

#     def _get_feed_dict(self, states, using_duplicate_q=True, actions=None):
#         if using_duplicate_q:
#             feed_dict = {
#                 self.states_feed_duplicate: states,
#             }
#             for r in self.recommenders:
#                 feed_dict[r.states_feed] = states
#         else:
#             feed_dict = {
#                 self.states_feed: states,
#                 self.actions_feed: actions
#             }
#         return feed_dict

#     def get_recommendations(self, states):
#         feed_dict = self._get_feed_dict(states)
#         actions, qs = self.session.run([self.recommenders_output, self.q_duplicate], feed_dict=feed_dict)
#         actions = np.reshape(actions, newshape=[len(self.recommenders), len(states), -1])
#         qs = np.reshape(qs, newshape=[len(self.recommenders), len(states)])
#         return actions, qs

#     def get_q(self, states, actions):
#         feed_dict = self._get_feed_dict(states, False, actions)
#         return self.session.run(self.q, feed_dict=feed_dict)

#     def train_recommenders(self, states):
#         feed_dict = self._get_feed_dict(states)
#         return self.session.run([self.av_q, self.train_recommenders_op], feed_dict=feed_dict)
        
#     def train_q(self, states, actions, target_qs):
#         feed_dict = self._get_feed_dict(states, False, actions)
#         feed_dict[self.target_q_feed] = target_qs
#         return self.session.run([self.mse, self.train_q_op], feed_dict=feed_dict)

#     def update_duplicate_q(self):
#         self.session.run(self.update_duplicate_q_op)
    

class Experience:
    def __init__(self, state, action, reward, done, info, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
        self.next_state = next_state

class ExperienceBuffer:
    def __init__(self, length=1e6):
        self.buffer = [] # type: List[Experience]
        self.buffer_length = length

    def __len__(self):
        return len(self.buffer)

    def add(self, exp: Experience):
        self.buffer.append(exp)
        if len(self.buffer) > self.buffer_length:
            self.buffer.pop(0)

    def random_experiences(self, count):
        indices = np.random.randint(0, len(self.buffer), size=count)
        for i in indices:
            yield self.buffer[i]

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class NormalNoise:
    def __init__(self, mu, sigma=0.2):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return self.mu + self.sigma * np.random.standard_normal(size=self.mu.shape)


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
        self.n_ambs = 18
        self.n_bases = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(0,1, shape=[self.n_bases])
        
    def step(self, action):
        print(np.round(self.n_ambs * action, decimals=2))
        obs,r,d,_ = super().step(action)
        r = r/200
        return obs, r, d, _

def test(sess):
    a = Actor(sess, 'actor', [5], [2], ob_dtype='float32', q_lr=1e-3, a_lr=1e-4, use_layer_norm=use_layer_norm, tau=tau)
    sess.run(tf.global_variables_initializer())
    a.update_target_networks()
    s = [[0.1,0.2,0.1,0.5,-0.1]]

    # print('Loading model')
    # a.load('/home/abhinav/models/{0}'.format(a.name))

    print('initial a&q on s={0}'.format(s[0]))
    print(a.get_actions_and_q(s))

    print('\nTraining q...')
    for i in range(500):
        _, mse = a.train_q([s[0]]*9, actions=[[-1,-1],[1,1],[-1,1],[1,-1],[-0.5, 0],[0,-0.5],[0,0.5],[1,0],[0.5,0]], target_q=[0,0,0,0,0.5,0.5,0.5,0.5,1])
        if i % 100 == 0:
            print('mse: {0}'.format(mse))

    print('After training q values:')
    print(a.get_q([s[0]]*9, [[-1,-1],[1,1],[-1,1],[1,-1],[-0.5, 0],[0,-0.5],[0,0.5],[1,0],[0.5,0]]))

    print('old a')
    print(a.get_actions_and_q(s))
    print('\nTraining a...')
    for i in range(500):
        a.train_a(s)
    print('New a & q:')
    print(a.get_actions_and_q(s))
    
    print('confirming target network gives same values:')
    a.update_target_networks()
    print(a.get_target_actions_and_q(s))

    print('Saving network')
    a.save('models/test1/{0}'.format(a.name))

def test2(sess):
    a = Actor(sess, 'actor', [5], [2], ob_dtype='float32', q_lr=1e-3, a_lr=1e-4, use_layer_norm=use_layer_norm, tau=tau)
    sess.run(tf.global_variables_initializer())
    a.update_target_networks()
    s = [[0.1,0.2,0.1,0.5,-0.1]]

    print('initial a&q on s={0}'.format(s[0]))
    print(a.get_actions_and_q(s))

    print('\nTraining q...')
    for i in range(500):
        _, mse = a.train_q([s[0]]*3, actions=[[0, 1],[1,0],[0.4,0.6]], target_q=[0.9,0.8,1])
        if i % 100 == 0:
            print('mse: {0}'.format(mse))

    print('After training q values:')
    print(a.get_q([s[0]]*3,[[0, 1],[1,0],[0.4,0.6]]))

    print('old a')
    print(a.get_actions_and_q(s))
    print('\nTraining a...')
    for i in range(500):
        a.train_a(s)
    print('New a & q:')
    print(a.get_actions_and_q(s))
    
    print('confirming target network gives same values:')
    a.update_target_networks()
    print(a.get_target_actions_and_q(s))

def test_envs():
    num_envs = 4
    def make_env(rank):
        def _thunk():
            print('Making env {0}'.format(env_id))
            env = gym.make(env_id)
            env.seed(seed + rank)
            #env = bench.Monitor(env, logger.get_dir() and 
            #    os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)), allow_early_resets = (policy in ('greedy', 'ga', 'rat')))
            #gym.logger.setLevel(logging.WARN)
            #return ObsExpandWrapper(env)
            #return NoopFrameskipWrapper(ObsExpandWrapper(env))
            return env
        return _thunk
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    Rs = []
    for ep in range(3):
        obs = env.reset()
        d = False
        R = np.zeros([num_envs])
        while not d:
            obs, r, d, info = env.step([np.array([4,4,2,0,3,5])/18]*num_envs)
            R += r
            d = d[0]
        Rs.append(R)
        print(R)
    env.close()


def normalize(a):
    a = np.clip(a, 0, 1)
    a = a + 1e-6
    a = a / np.sum(a)
    return a

def test_actor_on_env(sess, learning = False, actor=None):
    np.random.seed(seed)
    env = gym.make(env_id) # type: gym.Env
    for W in wrappers: env = W(env) # type: gym.Wrapper
    if actor is None:
        actor = Actor(sess, 'actor', env.observation_space.shape, env.action_space.shape, ob_dtype='float32', q_lr=1e-3, a_lr=1e-4, use_layer_norm=use_layer_norm, tau=tau)
        sess.run(tf.global_variables_initializer())
    actor.update_target_networks()
    if learning:
        experience_buffer = ExperienceBuffer(length=replay_memory_length)
        noise = Noise_type(mu=np.zeros(env.action_space.shape), sigma=exploration_sigma)
    def train():
        count = 1000 if f == exploration_period else 1
        for c in range(count):
            mb = list(experience_buffer.random_experiences(count=minibatch_size)) # type: List[Experience]
            s, a, s_next, r, d = [e.state for e in mb], [e.action for e in mb], [e.next_state for e in mb], np.asarray([e.reward for e in mb]), np.asarray([int(e.done) for e in mb])
            r += (1-d) * gamma * actor.get_target_actions_and_q(s_next)[1]
            _, mse = actor.train_q(s, r, a)
            actor.soft_update_target_networks()
        _, av_q = actor.train_a(s)
        if f == exploration_period:
            for c in range(count):
                mb = list(experience_buffer.random_experiences(count=minibatch_size)) # type: List[Experience]
                s = [e.state for e in mb]
                _, av_q = actor.train_a(s)
        if f % 100 == 0:
            print('mse: {0}\tav_q:{1}'.format(mse, av_q))
    def act(obs):
        if not learning:
            return actor.get_actions_and_q([obs])[0][0]
        if f < exploration_period:
            a = env.action_space.sample()
            if 'ERS' in env_id: a = normalize(a)
        else:
            a,q = actor.get_actions_and_q([obs])
            a,q = a[0],q[0]
            if 'ERS' in env_id: print('a=\t\t{0}\tq= {1}'.format(a,q))
        a += noise()
        a = normalize(a) if 'ERS' in env_id else np.clip(a, -1, 1)
        return a
    Rs, f = [], 0
    env.seed(learning_env_seed if learning else test_env_seed)
    for ep in range(learning_episodes if learning else test_episodes):
        obs, d, R, l = env.reset(), False, 0, 0
        while not d:
            if learning and f >= exploration_period and f % 4 == 0: train()
            a = act(obs)
            obs_, r, d, _ = env.step(a)
            if learning: experience_buffer.add(Experience(obs, a, r, d, _, obs_))
            obs, R, f, l = obs_, R+r, f+1, l+1
        Rs.append(R)
        av = np.average(Rs[-10:])
        print('Episode {0}:\tReward: {1}\tLength: {2}\tAv_R: {3}'.format(ep, R, l, av))
        if learning and env_id=='CartPole-v1' and av > 490: break
    env.close()
    print('Average reward per episode: {0}'.format(np.average(Rs)))
    return actor

if __name__ == '__main__':
    config = tf.ConfigProto(device_count = {'GPU': 0})
    env_id = sys.argv[1] if len(sys.argv) > 1 else 'pyERSEnv-ca-30-v3'
    print('env_id: ' + env_id)
    wrappers = [ERSEnvWrapper] if 'ERS' in env_id else [CartPoleWrapper]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print('Seed: {0}'.format(seed))
    np.random.seed(seed)
    minibatch_size = 64
    tau = 0.01
    gamma = 0.99
    exploration_period = 1000
    replay_memory_length = 100000
    exploration_sigma = 0.05 if 'ERS' in env_id else 0.2
    Noise_type = OrnsteinUhlenbeckActionNoise
    learning_env_seed = seed
    learning_episodes=1000
    test_env_seed = 0
    test_episodes = 100
    use_layer_norm = True
    nn_size = [400, 300]
    #with tf.Session(config=config) as sess: test(sess)
    #test_envs()
    with tf.Session(config=config) as sess:
        print('Training actor. seed={0}. learning_env_seed={1}'.format(seed, learning_env_seed))
        actor = test_actor_on_env(sess, True)
        actor.save('models/{0}/{1}/model_{2}'.format(env_id, seed, actor.name))
        print('Testing actor. test_env_seed={0}'.format(test_env_seed))
        test_actor_on_env(sess, False, actor=actor)
        print('Testing done. Seeds were seed={0}. learning_env_seed={1}. test_env_seed={2}'.format(seed, learning_env_seed, test_env_seed))
        print('-------------------------------------------------\n')

# env = DiscreteToContinousWrapper(gym.make('CartPole-v1'))
# env = CartPoleWrapper(gym.make('CartPole-v1'))
# rat = RAT(env, n_recommenders=4, seed=0, gamma=0.99, experience_buffer_length=10000, dup_q_update_interval=32, minibatch_size=32, timesteps=1e7, ob_dtype='float32', learning_rate=1e-3, render=False)

# 
# env = ERSEnvWrapper(gym.make('pyERSEnv-ca-dynamic-v3'))
# eval_env = ERSEnvWrapper(gym.make('pyERSEnv-ca-dynamic-v3'))
# env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "{}.monitor.json".format(0)), allow_early_resets = True, log_frames=False)
# gym.logger.setLevel(logging.WARN)
# rat = RAT(env, eval_env, n_recommenders=6, seed=0, gamma=1, experience_buffer_length=50000, exploration_period=48*100, dup_q_update_interval=500, 
#         update_interval=4, minibatch_size=64, pretrain_trainer=True, pretraining_steps=500, timesteps=1e7, ob_dtype='float32', learning_rate=5e-4, render=False)
# rat.epsilon_anneal = 48*500 # 100 episodes
# rat.epsilon_final = 0.2
# rat.use_beam_search = True
# rat.trainer_training_steps = 2
# rat.recommenders_training_steps = 1
# rat.beam_selection_interval = 100
# rat.evaluation_envs_count = 8
# rat.learn()
# plt.plot(rat.rewards)
# plt.show()

#eval code:
# alloc = [4,3,4,0,3,4]
# a = 2*np.array(alloc)/18 - 1
# env.seed(0)
# env.reset()
# R = []
# ep_r = 0
# while len(R) < 100:
#     obs, r, d, _ = env.step(a)
#     ep_r += r
#     if d:
#         print(ep_r)
#         R.append(ep_r)
#         ep_r = 0
#         obs = env.reset()

# print('Av score: {0}'.format(np.average(R)))