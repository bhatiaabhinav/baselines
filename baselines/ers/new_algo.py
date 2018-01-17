import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc


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
        self.buffer = []  # type: List[Experience]
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

    def random_trajectory(self, trajectory_length=50):
        index_end = np.random.randint(trajectory_length, len(self.buffer))
        index_start = index_end - trajectory_length
        return self.buffer[index_start:index_end]


class Model:
    def __init__(self, session: tf.Session, name, ob_shape, ac_shape, gamma=0.99, ob_dtype='float32', delta_R_lr=1e-3, a_lr=1e-3, n_lr=1e-3, v_lr=1e-3, goals_lr=1e-3, use_layer_norm=True, tau=0.001):
        with tf.variable_scope(name):
            self.states_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
            self.connect_graphs = tf.placeholder(dtype=tf.bool)
            with tf.variable_scope('goals'):
                h1 = fc(self.states_feed, 'h1', 128)
                h2 = fc(h1, 'h2', 64)
                self.goal_delta = fc(h2, 'goal', ob_shape[0], act=tf.nn.tanh)
                self.goals = self.goal_delta + self.states_feed
            with tf.variable_scope('n'):
                self.n_goals_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
                goals_feed = tf.case([(self.connect_graphs, lambda:self.goals)],
                                     default=lambda: self.n_goals_feed)
                feed = tf.concat([self.states_feed, goals_feed], axis=-1)
                h1 = fc(feed, 'h1', 128)
                h2 = fc(h1, 'h2', 64)
                self.n = fc(h2, 'n', 1, act=lambda x: x)[:, 0]
            with tf.variable_scope('delta_R'):
                self.delta_R_goals_feed = tf.placeholder(
                    dtype=ob_dtype, shape=[None] + list(ob_shape))
                goals_feed = tf.case([(self.connect_graphs, lambda:self.goals)],
                                     default=lambda: self.delta_R_goals_feed)
                feed = tf.concat([self.states_feed, goals_feed], axis=-1)
                h1 = fc(feed, 'h1', 128)
                h2 = fc(h1, 'h2', 64)
                self.delta_R = fc(h2, 'delta_R', 1, lambda x: x)[:, 0]
            with tf.variable_scope('a'):
                self.a_goals_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
                goals_feed = tf.case([(self.connect_graphs, lambda:self.goals)],
                                     default=lambda: self.a_goals_feed)
                feed = tf.concat([self.states_feed, goals_feed], axis=-1)
                h1 = fc(feed, 'h1', 128)
                h2 = fc(h1, 'h2', 64)
                self.a = fc(h2, 'a', ac_shape[0], lambda x: x)
            with tf.variable_scope('v'):
                self.v_states_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ob_shape))
                states_feed = tf.case([(self.connect_graphs, lambda:self.goals)],
                                      default=lambda: self.v_states_feed)
                h1 = fc(states_feed, 'h1', 128)
                h2 = fc(h1, 'h2', 64)
                self.v = fc(h2, 'v', 1, act=lambda x: x)[:, 0]

        dummy_states = [np.zeros(ob_shape)]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # setup training a
        self.a_training_a_feed = tf.placeholder(dtype=ob_dtype, shape=[None] + list(ac_shape))
        self.a_training_error = tf.reduce_mean(tf.square(self.a - self.a_training_a_feed) / 2)
        optimizer = tf.train.AdamOptimizer(learning_rate=a_lr)
        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/a'.format(name))
        with tf.control_dependencies(update_ops):
            self.a_training_op = optimizer.minimize(self.a_training_error, var_list=a_vars)

        # setup training n
        self.n_training_n_feed = tf.placeholder(dtype=tf.float32, shape=[None])
        self.n_training_error = tf.reduce_mean(tf.square(self.n - self.n_training_n_feed) / 2)
        optimizer = tf.train.AdamOptimizer(learning_rate=n_lr)
        n_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/n'.format(name))
        with tf.control_dependencies(update_ops):
            self.n_training_op = optimizer.minimize(self.n_training_error, var_list=n_vars)

        # setup training delta_R
        self.delta_R_training_delta_R_feed = tf.placeholder(dtype=tf.float32, shape=[None])
        self.delta_R_training_error = tf.reduce_mean(
            tf.square(self.delta_R - self.delta_R_training_delta_R_feed) / 2)
        optimizer = tf.train.AdamOptimizer(learning_rate=delta_R_lr)
        delta_R_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/delta_R'.format(name))
        with tf.control_dependencies(update_ops):
            self.delta_R_training_op = optimizer.minimize(
                self.delta_R_training_error, var_list=delta_R_vars)

        # setup training v
        self.v_training_v_feed = tf.placeholder(dtype=tf.float32, shape=[None])
        self.v_training_error = tf.reduce_mean(tf.square(self.v - self.v_training_v_feed) / 2)
        optimizer = tf.train.AdamOptimizer(learning_rate=v_lr)
        v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{0}/v'.format(name))
        with tf.control_dependencies(update_ops):
            self.v_training_op = optimizer.minimize(self.v_training_error, var_list=v_vars)

        # setup training goals
        self.objective_fn = self.delta_R + (gamma**self.n) * self.v
        optimizer = tf.train.AdamOptimizer(learning_rate=goals_lr)
        goals_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='{0}/goals'.format(name))
        with tf.control_dependencies(update_ops):
            self.goals_training_op = optimizer.minimize(-self.objective_fn, var_list=goals_vars)

        def get_v(states):
            return session.run(self.v, feed_dict={self.v_states_feed: states, self.connect_graphs: False, self.states_feed: dummy_states})

        def get_n(states, goals):
            return session.run(self.n, feed_dict={self.states_feed: states, self.n_goals_feed: goals, self.connect_graphs: False})

        def get_delta_R(states, goals):
            return session.run(self.delta_R, feed_dict={self.states_feed: states, self.delta_R_goals_feed: goals, self.connect_graphs: False})

        def get_a(states, goals):
            return session.run(self.a, feed_dict={self.states_feed: states, self.a_goals_feed: goals, self.connect_graphs: False})

        def get_goals_and_a(states):
            return session.run([self.goals, self.a], feed_dict={self.states_feed: states, self.a_goals_feed: dummy_states, self.connect_graphs: True})

        def get_goals_a_v_n_delta_R(states):
            return session.run([self.goals, self.a, self.v, self.n, self.delta_R], feed_dict={
                self.states_feed: states,
                self.a_goals_feed: dummy_states,
                self.v_states_feed: dummy_states,
                self.n_goals_feed: dummy_states,
                self.delta_R_goals_feed: dummy_states,
                self.connect_graphs: True
            })

        def train_a(states, goals, actions, epochs=1):
            for e in range(epochs):
                _, mse = session.run([self.a_training_op, self.a_training_error], feed_dict={
                    self.states_feed: states,
                    self.connect_graphs: False,
                    self.a_goals_feed: goals,
                    self.a_training_a_feed: actions
                })
            return _, mse

        def train_n(states, goals, target_n, epochs=1):
            for e in range(epochs):
                _, mse = session.run([self.n_training_op, self.n_training_error], feed_dict={
                    self.states_feed: states,
                    self.connect_graphs: False,
                    self.n_goals_feed: goals,
                    self.n_training_n_feed: target_n
                })
            return _, mse

        def train_delta_R(states, goals, target_delta_R, epochs=1):
            for e in range(epochs):
                _, mse = session.run([self.delta_R_training_op, self.delta_R_training_error], feed_dict={
                    self.states_feed: states,
                    self.connect_graphs: False,
                    self.delta_R_goals_feed: goals,
                    self.delta_R_training_delta_R_feed: target_delta_R
                })
            return _, mse

        def train_v(states, target_v, epochs=1):
            for e in range(epochs):
                _, mse = session.run([self.v_training_op, self.v_training_error], feed_dict={
                    self.states_feed: dummy_states,
                    self.connect_graphs: False,
                    self.v_states_feed: states,
                    self.v_training_v_feed: target_v
                })
            return _, mse

        def train_goals(states, epochs=1):
            for e in range(epochs):
                _, mse = session.run([self.goals_training_op, self.objective_fn], feed_dict={
                    self.states_feed: states,
                    self.connect_graphs: True,
                    self.delta_R_goals_feed: dummy_states,
                    self.n_goals_feed: dummy_states,
                    self.v_states_feed: dummy_states
                })
            return _, mse

        self.get_v = get_v
        self.get_n = get_n
        self.get_delta_R = get_delta_R
        self.get_a = get_a
        self.get_goals_and_a = get_goals_and_a
        self.get_goals_a_v_n_delta_R = get_goals_a_v_n_delta_R
        self.train_a = train_a
        self.train_n = train_n
        self.train_delta_R = train_delta_R
        self.train_v = train_v
        self.train_goals = train_goals


def test(sess: tf.Session):
    model = Model(sess, 'main', [4], [2])
    sess.run(tf.global_variables_initializer())
    s = [0, 0.5, 1, 0.5]
    g = [0, 1, 1.5, 0.5]
    a = [0.3, 0.7]
    model.train_v([s, g], [10, 20], epochs=100)
    print(model.get_v([s]))
    model.train_n([s], [g], [3], epochs=100)
    print(model.get_n([s], [g]))
    model.train_delta_R([s], [g], [5.5], epochs=100)
    print(model.get_delta_R([s], [g]))
    model.train_a([s], [g], [a], epochs=100)
    print(model.get_a([s], [g]))
    print(model.get_goals_a_v_n_delta_R([s]))
    print(model.train_goals([s], epochs=100)[1])
    print(model.get_goals_a_v_n_delta_R([s]))


if __name__ == '__main__':
    np.random.seed(0)
    with tf.Session() as sess:
        test(sess)
