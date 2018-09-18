"""dqn.py: Implementation of Deepmind's DQN algorithm using Tensorflow and OpenAI gym.

original paper: https://www.nature.com/articles/nature14236

Usage:
    python3 dqn.py [env_id]
    env_id: optional. default='CartPole-v0'. refer https://gym.openai.com/envs/#classic_control to know about more available environments.

python dependencies:
    gym[classic_control], tensorflow, numpy

What has not been implemented yet:
    * Use of convolutional neural networks to learn atari environments from pixels.
    * Frame stacking: Stack last 4 observations to make the state more markovian, incase it is not markovian by default.
    * Code for saving and loading the network.
"""

__author__ = "Abhinav Bhatia"
__email__ = "bhatiaabhinav93@gmail.com"
__license__ = "gpl"
__version__ = "0.9.0"

import sys

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc


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
    '''A circular buffer to hold experiences'''

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
                self.buffer_length = self.size_in_bytes / sys.getsizeof(exp)
            self.buffer_length = int(self.buffer_length)
            print('Initializing experience buffer of length {0}'.format(
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

    def random_states(self, count):
        experiences = list(self.random_experiences(count))
        return [e.state for e in experiences]


class Context:
    gamma = 0.999
    env_id = 'CartPole-v0'
    env = None  # type: gym.Env
    session = None  # type: tf.Session
    layers = [64, 64]
    n_episodes = 10000
    experience_buffer_length = int(1e6)
    experience_buffer = None  # type: ExperienceBuffer
    minimum_experience = 100  # in frames, before learning can begin
    train_every = 4  # one sgd step every this many frames
    target_network_update_interval = 1000  # target network updated every this many frames
    minibatch_size = 32
    learning_rate = 1e-3
    epsilon = 0.05  # exploration rate in epislon-greedy action selection
    tau = 0.001  # ddpg style target network (soft) update step size
    ddpg_target_network_update_mode = True  # whether to use DDPG style target network update, or use original DQN style
    tensorboard_folder = 'dqn'


class Brain:
    def __init__(self, context: Context, name):
        self.name = name
        self.context = context
        with tf.variable_scope(name):
            self._create_network(context)

    def _create_network(self, context: Context, name='Q_network'):
        with tf.variable_scope(name):
            self._states_placeholder = tf.placeholder(dtype=context.env.observation_space.dtype, shape=[
                                                      None] + list(context.env.observation_space.shape), name='states')
            prev_layer = self._states_placeholder
            for layer_id, layer in enumerate(context.layers):
                h = tf.layers.dense(prev_layer, layer,
                                    activation=None, name='h{0}'.format(layer_id))
                h = tc.layers.layer_norm(h, scale=True, center=True)
                h = tf.nn.relu(h)
                prev_layer = h
            self._Q = tf.layers.dense(
                prev_layer, context.env.action_space.n, name='Q')
            self._best_action_id = tf.argmax(self._Q, axis=1, name='action')
            self._V = tf.reduce_max(self._Q, axis=1, name='V')

    def get_Q(self, states):
        return self.context.session.run(self._Q, feed_dict={
            self._states_placeholder: states
        })

    def get_V(self, states):
        return self.context.session.run(self._V, feed_dict={
            self._states_placeholder: states
        })

    def get_action(self, states):
        return self.context.session.run(self._best_action_id, feed_dict={
            self._states_placeholder: states
        })

    def get_vars(self):
        my_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='{0}/Q_network'.format(self.name))
        return my_vars

    def setup_copy_to(self, other_brain, name='copy_brain_ops'):
        with tf.variable_scope(name):
            other_brain = other_brain  # type: Brain
            my_vars = self.get_vars()
            other_brain_vars = other_brain.get_vars()
            copy_ops = []
            for my_var, other_var in zip(my_vars, other_brain_vars):
                copy_op = tf.assign(other_var, my_var)
                copy_ops.append(copy_op)
            return copy_ops

    def setup_soft_copy_to(self, other_brain, name='soft_copy_brain_ops'):
        with tf.variable_scope(name):
            other_brain = other_brain  # type: Brain
            my_vars = self.get_vars()
            other_brain_vars = other_brain.get_vars()
            assert len(my_vars) == len(
                other_brain_vars), "Something is wrong! Both brains should have same number of vars"
            copy_ops = []
            for my_var, other_var in zip(my_vars, other_brain_vars):
                copy_op = tf.assign(
                    other_var, my_var * self.context.tau + (1 - self.context.tau) * other_var)
                copy_ops.append(copy_op)
            return copy_ops

    def setup_training(self, name='train_Q_network'):
        with tf.variable_scope(name):
            context = self.context
            self._desired_Q_placeholder = tf.placeholder(
                dtype='float32', shape=[None, context.env.action_space.n])
            error = tf.square(self._Q - self._desired_Q_placeholder)
            self._mse = tf.reduce_mean(error)
            optimizer = tf.train.AdamOptimizer(context.learning_rate)
            self._sgd_step = optimizer.minimize(
                self._mse, var_list=self.get_vars())

    def train(self, mb_states, mb_desiredQ):
        return self.context.session.run([self._sgd_step, self._mse], feed_dict={
            self._states_placeholder: mb_states,
            self._desired_Q_placeholder: mb_desiredQ
        })


def dqn(context: Context):
    env = context.env = gym.make(context.env_id)  # type: gym.Env
    main_brain = Brain(context, 'main_brain')
    main_brain.setup_training('main_brain/train_Q_network')
    target_brain = Brain(context, 'target_brain')
    target_brain_update_op = main_brain.setup_copy_to(
        target_brain, 'target_brain_update_op')
    target_brain_soft_update_op = main_brain.setup_soft_copy_to(
        target_brain, 'target_brain_soft_update_op')

    with tf.Session() as session:
        context.session = session
        tf.summary.FileWriter(context.tensorboard_folder, session.graph)
        session.run(tf.global_variables_initializer())
        session.run(target_brain_update_op)
        context.experience_buffer = ExperienceBuffer(
            length=context.experience_buffer_length)
        episode_rewards, episode_lengths, f = [], [], 0
        for episode_id in range(context.n_episodes):
            R, L, done = 0, 0, False
            obs = env.reset()
            while not done:
                if np.random.random() > context.epsilon:
                    action = main_brain.get_action([obs])[0]
                else:
                    action = context.env.action_space.sample()
                obs1, r, done, info = env.step(action)
                experience = Experience(obs, action, r, done, info, obs1)
                context.experience_buffer.add(experience)
                # env.render()

                # let's train:
                if f % context.train_every == 0 and f > context.minimum_experience:
                    mb_exps = list(context.experience_buffer.random_experiences(
                        context.minibatch_size))
                    mb_states = [exp.state for exp in mb_exps]
                    mb_actions = [exp.action for exp in mb_exps]
                    mb_rewards = np.asarray([exp.reward for exp in mb_exps])
                    mb_dones = np.asarray([int(exp.done) for exp in mb_exps])
                    # The bug was in the following line. Had written [exp.state for exp in mb_exps]
                    mb_states1 = [exp.next_state for exp in mb_exps]
                    mb_states1_V = target_brain.get_V(mb_states1)
                    mb_gammas = (1 - mb_dones) * context.gamma
                    mb_desired_Q = main_brain.get_Q(mb_states)
                    for exp_id in range(context.minibatch_size):
                        mb_desired_Q[exp_id, mb_actions[exp_id]] = mb_rewards[exp_id] + \
                            mb_gammas[exp_id] * mb_states1_V[exp_id]
                    _, mse = main_brain.train(mb_states, mb_desired_Q)

                # update target network:
                if not context.ddpg_target_network_update_mode:
                    if f % context.target_network_update_interval == 0:
                        session.run(target_brain_update_op)
                        print('Target network updated')
                else:
                    session.run(target_brain_soft_update_op)

                obs = obs1
                R += r
                L += 1
                f += 1

            episode_rewards.append(R)
            episode_lengths.append(L)
            average_reward = np.mean(episode_rewards[-100:])
            print('Episode: {0}\tReward: {1}\tAverage Reward: {2}'.format(
                episode_id, R, average_reward))


if __name__ == '__main__':
    context = Context()
    if len(sys.argv) > 1:
        context.env_id = sys.argv[1]
    dqn(context)
