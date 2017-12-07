from typing import List
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc
import gym
import matplotlib.pyplot as plt
from baselines import logger
import os.path
import logging
from baselines import bench

class Recommender:

    def __init__(self, session: tf.Session, name, ob_shape, ac_shape, ob_dtype='float32'):
        assert len(ac_shape) == 1
        with tf.variable_scope('recommenders'):
            with tf.variable_scope(name):
                self.states_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
                h1 = fc(self.states_feed, 'fc1', nh=48, init_scale=np.sqrt(2))
                h2 = fc(h1, 'fc2', nh=24, init_scale=np.sqrt(2))
                h3 = fc(h2, 'fc3', nh=12, init_scale=np.sqrt(2))
                self.recommendation = fc(h3, 'reco', nh=ac_shape[0], act=tf.nn.sigmoid)
        self.scores = []
        self.age = 0
        self.session = session
        self.name = name
        self.get_vars_op = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'recommenders/{0}'.format(name))
        self.vars_feed_holder = []
        self.set_vars_op = []
        for v in self.get_vars_op:
            v_feed = tf.placeholder(dtype='float32', shape=v.shape)
            self.vars_feed_holder.append(v_feed)
            self.set_vars_op.append(v.assign(v_feed))

    def get_recommendations(self, states):
        return self.session.run(self.recommendation, feed_dict={self.states_feed: states})

    def get_recommendations_tensor(self):
        return self.recommendation

    def get_vars(self):
        return self.session.run(self.get_vars_op)

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

    def add_score(self, score):
        self.scores.append(score)

    def average_score(self):
        return np.average(self.scores)

    def increment_age(self, amt=1):
        self.age += amt

    def reset_scores(self):
        self.scores.clear()
    
    def reset_age(self):
        self.age = 0


class ActorTrainer:

    def __init__(self, session: tf.Session, recommenders: List[Recommender], ob_shape, ac_shape, ob_dtype='float32', learning_rate=1e-3):
        # build the graph
        self.recommenders = recommenders
        self.session = session
        self.ac_shape = ac_shape
        self.ob_shape = ob_shape
        with tf.variable_scope('actor_trainer'):
            for scope in ['q', 'q_duplicate']:
                with tf.variable_scope(scope):
                    if scope == 'q_duplicate':
                        self.states_feed_duplicate = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
                        self.recommenders_output = tf.concat([r.get_recommendations_tensor() for r in recommenders], axis=0)
                        states = tf.concat([self.states_feed_duplicate]*len(recommenders), axis=0)
                        actions = self.recommenders_output
                    else:
                        self.states_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
                        self.actions_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ac_shape))
                        states = self.states_feed
                        actions = self.actions_feed
           
                    # this will work only if state is single dimensional as well:
                    inp = tf.concat((states, actions), axis=-1)
                    h1 = fc(inp, 'fc1', nh=64, init_scale=np.sqrt(2))
                    h2 = fc(h1, 'fc2', nh=32, init_scale=np.sqrt(2))
                    h3 = fc(h2, 'fc3', nh=16, init_scale=np.sqrt(2))
                    if scope == 'q_duplicate':
                        self.q_duplicate = fc(h3, 'q', 1, act=lambda x: x)[:, 0]
                    else:
                        self.q = fc(h3, 'q', 1, act=lambda x: x)[:, 0]
        
        optimizer_q = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer_recommenders = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # for training the recommenders:
        recommenders_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='recommenders')
        self.av_q = tf.reduce_mean(self.q_duplicate)
        self.train_recommenders_op = optimizer_recommenders.minimize(-self.av_q, var_list=recommenders_vars)

        # for training the Q network:
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_trainer/q')
        self.target_q_feed = tf.placeholder(dtype='float32', shape=[None])
        #clipped_diff = tf.clip_by_value(self.q - self.target_q_feed, -1, 1)
        se = tf.square(self.q - self.target_q_feed)/2
        self.mse = tf.reduce_mean(se)
        self.train_q_op = optimizer_q.minimize(self.mse, var_list=q_vars)

        # for updating duplicate q:
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_trainer/q')
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor_trainer/q_duplicate')
        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        self.update_duplicate_q_op = op_holder

    def _get_feed_dict(self, states, using_duplicate_q=True, actions=None):
        if using_duplicate_q:
            feed_dict = {
                self.states_feed_duplicate: states,
            }
            for r in self.recommenders:
                feed_dict[r.states_feed] = states
        else:
            feed_dict = {
                self.states_feed: states,
                self.actions_feed: actions
            }
        return feed_dict

    def get_recommendations(self, states):
        feed_dict = self._get_feed_dict(states)
        actions, qs = self.session.run([self.recommenders_output, self.q_duplicate], feed_dict=feed_dict)
        actions = np.reshape(actions, newshape=[len(self.recommenders), len(states), -1])
        qs = np.reshape(qs, newshape=[len(self.recommenders), len(states)])
        return actions, qs

    def get_q(self, states, actions):
        feed_dict = self._get_feed_dict(states, False, actions)
        return self.session.run(self.q, feed_dict=feed_dict)

    def train_recommenders(self, states):
        feed_dict = self._get_feed_dict(states)
        return self.session.run([self.av_q, self.train_recommenders_op], feed_dict=feed_dict)
        
    def train_q(self, states, actions, target_qs):
        feed_dict = self._get_feed_dict(states, False, actions)
        feed_dict[self.target_q_feed] = target_qs
        return self.session.run([self.mse, self.train_q_op], feed_dict=feed_dict)

    def update_duplicate_q(self):
        self.session.run(self.update_duplicate_q_op)
    

class Experience:
    def __init__(self, state, action, reward, done, info, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info
        self.next_state = next_state

class RAT:
    def __init__(self, env: gym.Env, eval_env: gym.Env, n_recommenders=4, seed=0, gamma=0.99, experience_buffer_length=10000, exploration_period=1000, dup_q_update_interval=500, update_interval=4, minibatch_size=32, pretrain_trainer=True, pretraining_steps=100, timesteps=1e7, ob_dtype='float32', learning_rate=1e-3, render=False):
        if n_recommenders <= 0:
            raise ValueError('There should be atleast one recommender')
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)

        self.ob_shape = env.observation_space.shape
        self.ac_shape = env.action_space.shape
        self.ob_dtype = ob_dtype
        self.learning_rate = learning_rate
        self.n_recommenders = n_recommenders
        self.experience_buffer_length = experience_buffer_length
        self.exploration_period = exploration_period
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.pretrain_trainer = pretrain_trainer
        self.pretraining_steps = pretraining_steps
        self.gamma = gamma
        self.timesteps = timesteps
        self.seed = seed
        self.render = render
        self.env = env
        self.eval_env = eval_env
        self.dup_q_update_interval = dup_q_update_interval
        self.experiences = [] # type: List[Experience]
        self.thinking_steps_before_acting = 0
        self.trainer_training_steps = 1
        self.recommenders_training_steps = 1
        self.use_beam_search = True
        self.beam_selection_interval = 1
        self.epsilon_final = 0.1
        self.epsilon_anneal = 2000
        self.update_count = 0
        self.evaluation_envs_count = 4

        self.recommenders = [Recommender(sess, 'r{0}'.format(i), self.ob_shape, self.ac_shape, self.ob_dtype) for i in range(n_recommenders)]
        self.actor_trainer = ActorTrainer(sess, self.recommenders, self.ob_shape, self.ac_shape, self.ob_dtype, self.learning_rate)
        sess.run(tf.global_variables_initializer())

        # for variation... let each recommender be mutation of another:
        for r, r_mut in zip(self.recommenders[:-1], self.recommenders[1:]):
            r.copy_to(r_mut, mutation_probability=0.5)
        

    def _get_action(self, state, best_action=True, epsilon=0.1):
        a, q = self._get_actions([state], best_actions=best_action, epsilon=epsilon)
        return a[0], q[0]

    def _get_actions(self, states, best_actions=True, epsilon=0.1):
        recos, qs = self.actor_trainer.get_recommendations(states)
        actions = []
        q = []
        for i in range(len(states)):
            best_action_index = np.argmax(qs[:,i])
            best_action = recos[best_action_index, i]
            best_q = qs[best_action_index, i]
            if best_actions:
                actions.append(best_action)
                q.append(best_q)
            else:
                #print('epsilon: {0}'.format(epsilon))
                rand = np.random.rand()
                if rand < 1 - epsilon:
                    actions.append(best_action)
                    q.append(best_q)
                elif rand < 1 - epsilon*epsilon:
                    random_action_index = np.random.randint(len(self.recommenders))
                    action = recos[random_action_index, i]
                    noisy_action = np.clip(action + np.random.standard_normal(size=np.shape(action))/10, 0, 1)
                    actions.append(noisy_action)
                    q.append(qs[random_action_index, i])
                    print('Noisy action')
                elif rand <= 1:
                    actions.append(self.env.action_space.sample())
                    q.append(0)
                    print('Random action')
                else:
                    raise RuntimeError('This should not happen. rand is {0}'.format(rand))
                for j in range(len(self.recommenders)):
                    self.recommenders[j].add_score(qs[j, i])
            
        return actions, q
    
    def _get_epsilon(self):
        if self.t < self.exploration_period:
            return 1
        if self.t >= (self.exploration_period + self.epsilon_anneal):
            return self.epsilon_final
        return 1 - (1 - self.epsilon_final) * (self.t - self.exploration_period)/(self.epsilon_anneal)

    def _add_to_experience(self, exp: Experience):
        self.experiences.append(exp)
        if len(self.experiences) > self.experience_buffer_length:
            self.experiences.pop(0)

    def _random_experiences(self, count):
        indices = np.random.randint(0, len(self.experiences) - 1, size=count)
        for i in indices:
            yield self.experiences[i]

    def _should_train_trainer(self):
        ans = self.t > self.exploration_period and self.t % self.update_interval == 0
        return ans

    def _should_train_recommenders(self):
        ans = self.t > self.exploration_period and self.t % self.update_interval == 0
        return ans

    def _should_select_recommenders(self):
        ans = self.use_beam_search and self.episodes % self.beam_selection_interval == 0
        if ans and self.n_recommenders % 2 != 0:
            raise RuntimeError('To do selection, number of recommenders should be even')
        return ans

    def _should_update_duplicate_q(self):
        return self.update_count % self.dup_q_update_interval == 0

    def _think_before_acting(self, state):
        for step in range(self.thinking_steps_before_acting):
            self.actor_trainer.train_recommenders([state])

    def _train_trainer(self, steps):
        for step in range(steps):
            exps = list(self._random_experiences(self.minibatch_size)) # type: List[Experience]
            next_states = [e.next_state for e in exps]
            # pass best_actions=False to do on-policy learning
            next_acts, next_qs = self._get_actions(next_states, best_actions=True)
            target_qs = [0] * self.minibatch_size
            for i in range(self.minibatch_size):
                target_qs[i] = exps[i].reward
                if not exps[i].done:
                    target_qs[i] += self.gamma * next_qs[i]
            loss, _ = self.actor_trainer.train_q([e.state for e in exps], [e.action for e in exps], target_qs)
            #print('loss: {0}'.format(loss))
            self.update_count += 1
            if self._should_update_duplicate_q():
                self.actor_trainer.update_duplicate_q()

    def _train_recommenders(self, steps):
        for step in range(steps):
            exps = self._random_experiences(self.minibatch_size)
            states = [e.state for e in exps]
            av_q, _ = self.actor_trainer.train_recommenders(states)
            #print('av_q: {0}'.format(av_q))

    def _eval_recommenders(self):
        base_seed = np.random.randint(0, 100)
        scores = []
        print('Evaluating recommenders on base_seed {0}'.format(base_seed))
        for reco in self.recommenders:
            #print('Reco #{0} :'.format(len(scores)))
            R = []
            for test_no in range(self.evaluation_envs_count):
                seed = base_seed + test_no
                self.eval_env.seed(seed)
                obs = self.eval_env.reset()
                ep_r = 0
                d = False
                while not d:
                    a = reco.get_recommendations([obs])[0]
                    obs, r, d, _ = self.eval_env.step(a)
                    ep_r += r
                #print(ep_r)
                R.append(ep_r)
            scores.append(np.average(R))
        print('Evalution scores: {0}'.format(scores))
        return np.array(scores)
        #return np.array([reco.average_score() for reco in self.recommenders])
            

    def _select_recommenders(self):
        p = self._eval_recommenders()
        p -= np.min(p)
        if np.max(p) > 0:
            p /= (np.max(p)/4)
        else:
            p = np.array([1]*len(p))
        p = np.exp(p)
        p /= np.sum(p)
        selected = np.random.choice(len(p), p=p, size=len(p)//2, replace=False)
        notSelected = list(filter(lambda i: i not in selected, range(len(p))))
        mutation_probs = np.linspace(0.1, 0.5, num=len(selected))
        #print('Selected recommenders:')
        #print(selected)
        #print('To be replaced recommenders:')
        #print(list(notSelected))
        #print('Mutation probs:')
        #print(mutation_probs)
        for s, ns, mp in zip(selected, notSelected, mutation_probs):
            #print(mp)
            self.recommenders[s].copy_to(self.recommenders[ns], mutation_probability=mp)
            self.recommenders[ns].reset_age()
        # for r in self.recommenders:
        #     r.reset_scores()

    def learn(self):
        #for i in range(100):
        #    self.actor_trainer.train_q([np.random.rand(self.ob_shape[0]), np.random.rand(self.ob_shape[0])], [[-2],[2]], [-10,-10])

        self.env.seed(self.seed)
        self.t = 0
        self.episodes = 0
        self.rewards = []
        for r in self.recommenders:
            r.reset_scores()
        while self.t < self.timesteps:
            if self._should_select_recommenders():
                self._select_recommenders()
            for r in self.recommenders:
                r.reset_scores()
            obs = self.env.reset()
            if self.render: self.env.render()
            R = 0
            d = False
            while not d:
                # general training
                if self.pretrain_trainer and self.t == self.exploration_period:
                    print('Exploration phase ended. Pretraining trainer:')
                    self._train_trainer(self.pretraining_steps)
                if self._should_train_trainer():
                    self._train_trainer(self.trainer_training_steps)
                if self._should_train_recommenders():
                    self._train_recommenders(self.recommenders_training_steps)
                
                self._think_before_acting(obs)
                a, q = self._get_action(obs, best_action=False, epsilon=self._get_epsilon())
                next_obs, r, d, info = self.env.step(a)
                self._add_to_experience(Experience(obs, a, r, d, info, next_obs))
                obs = next_obs
                R += r
                if self.render: self.env.render()
                self.t += 1
            
            for r in self.recommenders:
                r.increment_age()
            print('Scores of recommenders:')
            p = np.array([r.average_score() for r in self.recommenders])
            print(p)
            
            self.episodes += 1
            self.rewards.append(R)
            print('episode {0}:\t{1}'.format(self.episodes,R))
            if sum(self.rewards[-5:]) > (400*5):
                break


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
        self.env = env
        self.obs = None
        self.decision_interval = 30
        self.n_ambs = 18
        self.n_bases = 6
        
    def step(self, action):
        action = action / np.sum(action)
        print(self.n_ambs * action)
        rewards = []
        while len(rewards) < self.decision_interval:
            self.obs, r, d, info = super().step(action)
            rewards.append(r)
            if d:
                break
        return self.obs, np.sum(rewards), d, info

    def reset(self):
        self.obs = super().reset()
        return self.obs


if __name__ == '__main__':
    # config = tf.ConfigProto(device_count = {'GPU': 0})
    # sess = tf.Session(config=config)
    
    # r1 = Recommender(sess, 'rec1', [5], [2], ob_dtype='float32')
    # r2 = Recommender(sess, 'rec2', [5], [2], ob_dtype='float32')
    # at = ActorTrainer(sess, [r1, r2], [5], [2], ob_dtype='float32', learning_rate=1e-3)
    # sess.run(tf.global_variables_initializer())
    # at.update_duplicate_q()

    s = [[0.1,0.2,0.1,0.5,-0.1]]
    #print(r1.get_recommendations(s))
    #print(r2.get_recommendations(s))
    # print('Original recommendations:')
    # print(at.get_recommendations(s))

    # r1.copy_to(r2, mutation_probability=0.05)

    # print('\nTraining q...')
    # for i in range(100):
    #     at.train_q([s[0]]*9, [[0,0],[1,1],[0,1],[1,0],[0.5,0.25],[0.25,0.5],[0.5,0.75],[0.75,0.5],[0.5,0.5]], [0,0,0,0,0.5,0.5,0.5,0.5,1])

    # print('After training q values:')
    # print(at.get_q([s[0]]*9, [[0,0],[1,1],[0,1],[1,0],[0.5,0.25],[0.25,0.5],[0.5,0.75],[0.75,0.5],[0.5,0.5]]))

    # at.update_duplicate_q()
    # print('\nTraining recommenders...')
    # for i in range(20):
    #     at.train_recommenders(s)
    # print('New recommendations:')
    # print(at.get_recommendations(s))


np.random.seed(0)

# env = DiscreteToContinousWrapper(gym.make('CartPole-v1'))
# env = CartPoleWrapper(gym.make('CartPole-v1'))
# rat = RAT(env, n_recommenders=4, seed=0, gamma=0.99, experience_buffer_length=10000, dup_q_update_interval=32, minibatch_size=32, timesteps=1e7, ob_dtype='float32', learning_rate=1e-3, render=False)

import gym_ERSLE
env = ERSEnvWrapper(gym.make('pyERSEnv-ca-dynamic-v3'))
eval_env = ERSEnvWrapper(gym.make('pyERSEnv-ca-dynamic-v3'))
env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "{}.monitor.json".format(0)), allow_early_resets = True, log_frames=False)
gym.logger.setLevel(logging.WARN)
rat = RAT(env, eval_env, n_recommenders=6, seed=0, gamma=1, experience_buffer_length=50000, exploration_period=48*100, dup_q_update_interval=500, 
        update_interval=4, minibatch_size=64, pretrain_trainer=True, pretraining_steps=500, timesteps=1e7, ob_dtype='float32', learning_rate=5e-4, render=False)
rat.epsilon_anneal = 48*500 # 100 episodes
rat.epsilon_final = 0.2
rat.use_beam_search = True
rat.trainer_training_steps = 2
rat.recommenders_training_steps = 1
rat.beam_selection_interval = 100
rat.evaluation_envs_count = 8
rat.learn()
plt.plot(rat.rewards)
plt.show()

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