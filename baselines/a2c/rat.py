from typing import List
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc
import gym
import gym_ERSLE
import matplotlib.pyplot as plt
from baselines import logger
import os.path
import logging
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import sys

class Actor:

    def __init__(self, session: tf.Session, name, ob_shape, ac_shape, ob_dtype='float32', q_lr=1e-3, a_lr=1e-4, use_layer_norm=True, auto_update_target_network_interval=100):
        assert len(ac_shape) == 1
        self.session = session
        self.name = name
        self.ac_shape = ac_shape
        self.auto_update_target_network_interval = auto_update_target_network_interval
        self.updates = 0
        with tf.variable_scope(name):
            for scope in ['original', 'target']:
                with tf.variable_scope(scope):
                    with tf.variable_scope('s'):
                        states_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
                        # conv layers go here
                        states_flat = states_feed
                        states_flat = tf.layers.batch_normalization(states_flat)
                        s_h1 = fc(states_flat, 's_h1', nh=64, init_scale=np.sqrt(2))
                        if use_layer_norm:
                            s_h1 = tf.layers.batch_normalization(s_h1)
                        s = s_h1
                        if scope == 'target':
                            self.states_feed_target = states_feed
                        else:
                            self.states_feed = states_feed
                    with tf.variable_scope('a'):
                        # one hidden layer after state representation for actions:
                        a_h1 = fc(s, 'a_h1', nh=32, init_scale=np.sqrt(2))
                        if use_layer_norm:
                            a_h1 = tf.layers.batch_normalization(a_h1)
                        a = fc(a_h1, 'a', nh=ac_shape[0], act=tf.nn.tanh, init_scale=1e-3)
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
                        s_a_concat = tf.concat([s, a], axis=-1)
                        # one hidden layer after concating s,a
                        q_h1 = fc(s_a_concat, 'q_h1', nh=32, init_scale=np.sqrt(2))
                        if use_layer_norm:
                            q_h1 = tf.layers.batch_normalization(q_h1)
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
        self.q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/q'.format(name)) + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/original/s'.format(name))
        self.q_target_feed = tf.placeholder(dtype='float32', shape=[None])
        se = tf.square(self.q - self.q_target_feed)/2
        self.mse = tf.reduce_mean(se)
        
        with tf.control_dependencies(update_ops):
            self.train_q_op = optimizer_q.minimize(self.mse, var_list=self.q_vars)

        # for updating target network:
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/original'.format(name))
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{0}/target'.format(name))
        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        self.update_target_network_op = op_holder

    def reset(self):
        self.updates = 0
        self.update_target_network()

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
        results = self.session.run(ops, feed_dict=feed)
        self.updates += 1
        if self.auto_update_target_network_interval > 0 and self.updates % self.auto_update_target_network_interval == 0:
            self.update_target_network()
        return results
    
    def get_train_q_op_and_feed(self, states, target_q, actions=None):
        if actions is None:
            actions = [np.zeros(self.ac_shape)]
        return [self.train_q_op, self.mse], {self.states_feed:states, self.use_actions_feed: (actions is not None), self.actions_feed: actions, self.q_target_feed: target_q}
    
    def train_a(self, states):
        ops, feed = self.get_train_a_op_and_feed(states)
        results = self.session.run(ops, feed_dict=feed)
        self.updates += 1
        if self.auto_update_target_network_interval > 0 and self.updates % self.auto_update_target_network_interval == 0:
            self.update_target_network()
        return results
    
    def get_train_a_op_and_feed(self, states):
        return [self.train_a_op, self.av_q], {self.states_feed:states, self.use_actions_feed: False, self.actions_feed: [np.zeros(self.ac_shape)]}

    def get_target_actions_and_q(self, states):
        ops, feed = self.get_target_actions_and_q_op_and_feed(states)
        return self.session.run(ops, feed_dict=feed)

    def get_target_actions_and_q_op_and_feed(self, states):
        return [self.a_target, self.q_target], {self.states_feed_target: states}

    def update_target_network(self):
        print('updating target network')
        ops, feed = self.get_update_target_network_op_and_feed()
        self.session.run(ops, feed_dict=feed)

    def get_update_target_network_op_and_feed(self):
        return [self.update_target_network_op], {}

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
            self.experiences.pop(0)

    def random_experiences(self, count):
        indices = np.random.randint(0, len(self.buffer), size=count)
        for i in indices:
            yield self.buffer[i]

# class RAT:
#     def __init__(self, envs: List[gym.Env], eval_env: gym.Env, seed=0, gamma=0.99, experience_buffer_length=10000, exploration_period=1000, dup_q_update_interval=500, update_interval=4, minibatch_size=32, pretrain_trainer=True, pretraining_steps=100, timesteps=1e7, ob_dtype='float32', learning_rate=1e-3, render=False):        
#         config = tf.ConfigProto(device_count = {'GPU': 0})
#         sess = tf.Session(config=config)

#         self.ob_shape = env.observation_space.shape
#         self.ac_shape = env.action_space.shape
#         self.ob_dtype = ob_dtype
#         self.learning_rate = learning_rate
#         self.n_recommenders = len(envs)
#         self.experience_buffer_length = experience_buffer_length
#         self.exploration_period = exploration_period
#         self.update_interval = update_interval
#         self.minibatch_size = minibatch_size
#         self.pretrain_trainer = pretrain_trainer
#         self.pretraining_steps = pretraining_steps
#         self.gamma = gamma
#         self.timesteps = timesteps
#         self.seed = seed
#         self.render = render
#         self.env = env
#         self.eval_env = eval_env
#         self.dup_q_update_interval = dup_q_update_interval
#         self.experiences = [] # type: List[Experience]
#         self.thinking_steps_before_acting = 0
#         self.trainer_training_steps = 1
#         self.recommenders_training_steps = 1
#         self.use_beam_search = True
#         self.beam_selection_interval = 1
#         self.epsilon_final = 0.1
#         self.epsilon_anneal = 2000
#         self.update_count = 0
#         self.evaluation_envs_count = 4

#         self.recommenders = [Recommender(sess, 'r{0}'.format(i), self.ob_shape, self.ac_shape, self.ob_dtype) for i in range(n_recommenders)]
#         self.actor_trainer = ActorTrainer(sess, self.recommenders, self.ob_shape, self.ac_shape, self.ob_dtype, self.learning_rate)
#         sess.run(tf.global_variables_initializer())

#         # for variation... let each recommender be mutation of another:
#         for r, r_mut in zip(self.recommenders[:-1], self.recommenders[1:]):
#             r.copy_to(r_mut, mutation_probability=0.5)
        

#     def _get_action(self, state, best_action=True, epsilon=0.1):
#         a, q = self._get_actions([state], best_actions=best_action, epsilon=epsilon)
#         return a[0], q[0]

#     def _get_actions(self, states, best_actions=True, epsilon=0.1):
#         recos, qs = self.actor_trainer.get_recommendations(states)
#         actions = []
#         q = []
#         for i in range(len(states)):
#             best_action_index = np.argmax(qs[:,i])
#             best_action = recos[best_action_index, i]
#             best_q = qs[best_action_index, i]
#             if best_actions:
#                 actions.append(best_action)
#                 q.append(best_q)
#             else:
#                 #print('epsilon: {0}'.format(epsilon))
#                 rand = np.random.rand()
#                 if rand < 1 - epsilon:
#                     actions.append(best_action)
#                     q.append(best_q)
#                 elif rand < 1 - epsilon*epsilon:
#                     random_action_index = np.random.randint(len(self.recommenders))
#                     action = recos[random_action_index, i]
#                     noisy_action = np.clip(action + np.random.standard_normal(size=np.shape(action))/10, 0, 1)
#                     actions.append(noisy_action)
#                     q.append(qs[random_action_index, i])
#                     print('Noisy action')
#                 elif rand <= 1:
#                     actions.append(self.env.action_space.sample())
#                     q.append(0)
#                     print('Random action')
#                 else:
#                     raise RuntimeError('This should not happen. rand is {0}'.format(rand))
#                 for j in range(len(self.recommenders)):
#                     self.recommenders[j].add_score(qs[j, i])
            
#         return actions, q
    
#     def _get_epsilon(self):
#         if self.t < self.exploration_period:
#             return 1
#         if self.t >= (self.exploration_period + self.epsilon_anneal):
#             return self.epsilon_final
#         return 1 - (1 - self.epsilon_final) * (self.t - self.exploration_period)/(self.epsilon_anneal)



#     def _should_train_trainer(self):
#         ans = self.t > self.exploration_period and self.t % self.update_interval == 0
#         return ans

#     def _should_train_recommenders(self):
#         ans = self.t > self.exploration_period and self.t % self.update_interval == 0
#         return ans

#     def _should_select_recommenders(self):
#         ans = self.use_beam_search and self.episodes % self.beam_selection_interval == 0
#         if ans and self.n_recommenders % 2 != 0:
#             raise RuntimeError('To do selection, number of recommenders should be even')
#         return ans

#     def _should_update_duplicate_q(self):
#         return self.update_count % self.dup_q_update_interval == 0

#     def _think_before_acting(self, state):
#         for step in range(self.thinking_steps_before_acting):
#             self.actor_trainer.train_recommenders([state])

#     def _train_trainer(self, steps):
#         for step in range(steps):
#             exps = list(self._random_experiences(self.minibatch_size)) # type: List[Experience]
#             next_states = [e.next_state for e in exps]
#             # pass best_actions=False to do on-policy learning
#             next_acts, next_qs = self._get_actions(next_states, best_actions=True)
#             target_qs = [0] * self.minibatch_size
#             for i in range(self.minibatch_size):
#                 target_qs[i] = exps[i].reward
#                 if not exps[i].done:
#                     target_qs[i] += self.gamma * next_qs[i]
#             loss, _ = self.actor_trainer.train_q([e.state for e in exps], [e.action for e in exps], target_qs)
#             #print('loss: {0}'.format(loss))
#             self.update_count += 1
#             if self._should_update_duplicate_q():
#                 self.actor_trainer.update_duplicate_q()

#     def _train_recommenders(self, steps):
#         for step in range(steps):
#             exps = self._random_experiences(self.minibatch_size)
#             states = [e.state for e in exps]
#             av_q, _ = self.actor_trainer.train_recommenders(states)
#             #print('av_q: {0}'.format(av_q))

#     def _eval_recommenders(self):
#         base_seed = np.random.randint(0, 100)
#         scores = []
#         print('Evaluating recommenders on base_seed {0}'.format(base_seed))
#         for reco in self.recommenders:
#             #print('Reco #{0} :'.format(len(scores)))
#             R = []
#             for test_no in range(self.evaluation_envs_count):
#                 seed = base_seed + test_no
#                 self.eval_env.seed(seed)
#                 obs = self.eval_env.reset()
#                 ep_r = 0
#                 d = False
#                 while not d:
#                     a = reco.get_recommendations([obs])[0]
#                     obs, r, d, _ = self.eval_env.step(a)
#                     ep_r += r
#                 #print(ep_r)
#                 R.append(ep_r)
#             scores.append(np.average(R))
#         print('Evalution scores: {0}'.format(scores))
#         return np.array(scores)
#         #return np.array([reco.average_score() for reco in self.recommenders])
            

#     def _select_recommenders(self):
#         p = self._eval_recommenders()
#         p -= np.min(p)
#         if np.max(p) > 0:
#             p /= (np.max(p)/4)
#         else:
#             p = np.array([1]*len(p))
#         p = np.exp(p)
#         p /= np.sum(p)
#         selected = np.random.choice(len(p), p=p, size=len(p)//2, replace=False)
#         notSelected = list(filter(lambda i: i not in selected, range(len(p))))
#         mutation_probs = np.linspace(0.1, 0.5, num=len(selected))
#         #print('Selected recommenders:')
#         #print(selected)
#         #print('To be replaced recommenders:')
#         #print(list(notSelected))
#         #print('Mutation probs:')
#         #print(mutation_probs)
#         for s, ns, mp in zip(selected, notSelected, mutation_probs):
#             #print(mp)
#             self.recommenders[s].copy_to(self.recommenders[ns], mutation_probability=mp)
#             self.recommenders[ns].reset_age()
#         # for r in self.recommenders:
#         #     r.reset_scores()

#     def learn(self):
#         #for i in range(100):
#         #    self.actor_trainer.train_q([np.random.rand(self.ob_shape[0]), np.random.rand(self.ob_shape[0])], [[-2],[2]], [-10,-10])

#         self.env.seed(self.seed)
#         self.t = 0
#         self.episodes = 0
#         self.rewards = []
#         for r in self.recommenders:
#             r.reset_scores()
#         while self.t < self.timesteps:
#             if self._should_select_recommenders():
#                 self._select_recommenders()
#             for r in self.recommenders:
#                 r.reset_scores()
#             obs = self.env.reset()
#             if self.render: self.env.render()
#             R = 0
#             d = False
#             while not d:
#                 # general training
#                 if self.pretrain_trainer and self.t == self.exploration_period:
#                     print('Exploration phase ended. Pretraining trainer:')
#                     self._train_trainer(self.pretraining_steps)
#                 if self._should_train_trainer():
#                     self._train_trainer(self.trainer_training_steps)
#                 if self._should_train_recommenders():
#                     self._train_recommenders(self.recommenders_training_steps)
                
#                 self._think_before_acting(obs)
#                 a, q = self._get_action(obs, best_action=False, epsilon=self._get_epsilon())
#                 next_obs, r, d, info = self.env.step(a)
#                 self._add_to_experience(Experience(obs, a, r, d, info, next_obs))
#                 obs = next_obs
#                 R += r
#                 if self.render: self.env.render()
#                 self.t += 1
            
#             for r in self.recommenders:
#                 r.increment_age()
#             print('Scores of recommenders:')
#             p = np.array([r.average_score() for r in self.recommenders])
#             print(p)
            
#             self.episodes += 1
#             self.rewards.append(R)
#             print('episode {0}:\t{1}'.format(self.episodes,R))
#             if sum(self.rewards[-5:]) > (400*5):
#                 break


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
        self.n_bases = 6
        self.action_space = gym.spaces.Box(-1,1, shape=env.action_space.shape)
        
    def step(self, action):
        action = (action+1)/2
        action = np.clip((action + 1e-9) / (np.sum(action) + 1e-9), 0, 1)
        print(np.round(self.n_ambs * action, decimals=2))
        return super().step(action)

def test(sess):
    a = Actor(sess, 'r1', [5], [2], ob_dtype='float32', q_lr=1e-3, a_lr=1e-4)
    sess.run(tf.global_variables_initializer())
    a.reset()
    s = [[0.1,0.2,0.1,0.5,-0.1]]

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
    #a.update_target_network()
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


def test_actor_on_env(sess, learning = False, actor=None):
    np.random.seed(seed)
    env = gym.make(env_id) # type: gym.Env
    for W in wrappers: env = W(env) # type: gym.Wrapper
    if actor is None:
        actor = Actor(sess, 'r1', env.observation_space.shape, env.action_space.shape, ob_dtype='float32', q_lr=1e-3, a_lr=1e-4, use_layer_norm=use_layer_norm, auto_update_target_network_interval=100)
        sess.run(tf.global_variables_initializer())
    actor.reset()
    if learning: experience_buffer = ExperienceBuffer()
    def train():
        mb = list(experience_buffer.random_experiences(count=minibatch_size)) # type: List[Experience]
        s, a, s_next, r, d = [e.state for e in mb], [e.action for e in mb], [e.next_state for e in mb], np.asarray([e.reward for e in mb]), np.asarray([int(e.done) for e in mb])
        r += (1-d) * gamma * actor.get_target_actions_and_q(s_next)[1]
        _, mse = actor.train_q(s, r, a)
        _, av_q = actor.train_a(s)
        #print('mse: {0}\tav_q:{1}'.format(mse, av_q))
    def epsilon():
        if f < exploration_period: return 1
        if f >= (exploration_period + epsilon_anneal): return epsilon_final
        return 1 - (1 - epsilon_final) * (f - exploration_period)/(epsilon_anneal)
    def act(obs):
        if not learning:
            return actor.get_actions_and_q([obs])[0][0]
        if f < exploration_period: return env.action_space.sample()
        a = actor.get_actions_and_q([obs])[0][0]
        if np.random.rand() < epsilon():
            a = np.clip(a + 0.5 * np.random.standard_normal(size=np.shape(a)), -1, 1)
        return a
    Rs, f = [], 0
    env.seed(learning_env_seed if learning else test_env_seed)
    for ep in range(learning_episodes if learning else test_episodes):
        obs, d, R, l = env.reset(), False, 0, 0
        while not d:
            if learning and len(experience_buffer) > exploration_period and f % 4 == 0: train()
            a = act(obs)
            obs_, r, d, _ = env.step(a)
            if learning: experience_buffer.add(Experience(obs, a, r, d, _, obs_))
            obs, R, f, l = obs_, R+r, f+1, l+1
        Rs.append(R)
        av = np.average(Rs[-10:])
        print('Episode {0}:\tReward: {1}\tLength: {2}\tAv_R: {3}'.format(ep, R, l, av))
    env.close()
    print('Average reward per episode: {0}'.format(np.average(Rs)))
    return actor

if __name__ == '__main__':
    config = tf.ConfigProto(device_count = {'GPU': 0})
    env_id = sys.argv[1] if len(sys.argv) > 1 else 'pyERSEnv-ca-30-v3'
    print('env_id: ' + env_id)
    wrappers = [ERSEnvWrapper]
    #env_id = 'CartPole-v1'
    #wrappers = [CartPoleWrapper]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print('Seed: {0}'.format(seed))
    np.random.seed(seed)
    minibatch_size = 64
    gamma = 0.99
    exploration_period = 5000
    epsilon_anneal = 10000
    epsilon_final = 0.1
    learning_env_seed = seed
    learning_episodes=1000
    test_env_seed = 0
    test_episodes = 100
    #with tf.Session(config=config) as sess: test(sess)
    #test_envs()
    use_layer_norm = True
    with tf.Session(config=config) as sess:
        print('Training actor. seed={0}. learning_env_seed={1}'.format(seed, learning_env_seed))
        actor = test_actor_on_env(sess, True)
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