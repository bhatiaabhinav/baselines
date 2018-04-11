import os

import gym
import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.ers.addpg_ac_model import Actor
from baselines.ers.utils import StaffordRandFixedSum


def analyse_q(sess: tf.Session, env_id, wrappers, seed=0, test_env_seed=42, actor=None, load_path=None, **kwargs):
    np.random.seed(seed)
    env = gym.make(env_id)  # type: gym.Env
    for W in wrappers:
        env = W(env)  # type: gym.Wrapper
    N = 100000
    data_shape = [N] + list(env.action_space.shape)
    data_var = tf.Variable(tf.random_normal(
        data_shape), name='action_data')
    data_placeholder = tf.placeholder(
        dtype='float32', shape=data_shape, name='action_data_placeholder')
    data_load_op = tf.assign(data_var, data_placeholder)
    saver = tf.train.Saver()
    if actor is None:
        actor = Actor(sess, 'actor', env.observation_space.shape,
                      env.action_space.shape, softmax_actor='ERS' in env_id, **kwargs)
        sess.run(tf.global_variables_initializer())
    if load_path:
        try:
            actor.load(load_path)
            logger.log('model loaded')
        except Exception as ex:
            logger.log('Failed to load model. Reason = {0}'.format(
                ex), level=logger.ERROR)
    actor.update_target_networks()

    def Q(s, a):
        return actor.get_V_A_Q(s, a)[2]

    def argmax_Q(s):
        return actor.get_a_V_A_Q(s)[0]

    def analyse_q_vs_a_naive():
        obs = env.reset()
        done = False
        R = 0
        state_index = 0
        while not done:
            # for i in range(100):
            #     actor.train_a([obs])
            # take the default action
            a = argmax_Q([obs])[0]
            q = Q([obs], [a])[0]
            domain = [0, 1] if 'ERS' in env_id else [-1, 1]
            npoints = 500
            x = np.linspace(domain[0], domain[1], num=npoints)
            import matplotlib.pyplot as plt
            plt.clf()
            for idx in range(len(a)):
                # for this index: build the graph keeping other idxs of 'a' constant
                states = [obs] * npoints
                actions = np.asarray([a] * npoints)
                actions[:, idx] = x
                y = Q(states, actions)
                plt.plot(x, y, label='Q vs a[{0}]'.format(idx))
            plt.plot(a, [q] * len(a), 'rx', label='chosen action')
            plt.legend()
            plt.title('Q vs Action for State S{0}'.format(state_index))
            plt.ylabel('Learnt Q')
            plt.xlabel('Action components')
            plt.savefig(os.path.join(logger.get_dir(),
                                     'S{0:02d}_Q_vs_a.png'.format(state_index)))
            obs, r, done, _ = env.step(a)
            state_index += 1
            R += r
        logger.log('R: {0}'.format(R))

    def analyse_q_tb():
        obs = env.reset()
        data = StaffordRandFixedSum(env.action_space.shape[0], 1, N)
        logger.log('Got Random action data points')
        data_q = Q([obs] * N, data)
        logger.log('Got the Q values')
        sess.run(data_load_op, feed_dict={data_placeholder: data})
        logger.log('loaded the action data into tf variable')
        saver.save(sess, os.path.join(logger.get_dir(), 'model.ckpt'), 0)
        logger.log('saved the checkpoint')
        metadata_filename = os.path.join(logger.get_dir(), 'q_s0.tsv')
        projector_config_filename = os.path.join(
            logger.get_dir(), 'projector_config.ptxt')
        with open(projector_config_filename, 'w') as f:
            lines = "embeddings {{\ntensor_name: 'action_data'\nmetadata_path: '{metadata_filename}'\n}}".format(
                metadata_filename='$LOG_DIR/q_s0.tsv')
            f.write(lines)
        logger.log('wrote the config file')
        with open(metadata_filename, 'w') as f:
            # f.write('Q')
            for qvalue in data_q:
                f.write('{0}\n'.format(qvalue))
        logger.log('wrote the metadata file')

    # def analyse_q_scikit():
    #     obs = env.reset()
    #     data = StaffordRandFixedSum(env.action_space.shape[0], 1, N)
    #     logger.log('Got Random action data points')
    #     data_q = Q([obs] * N, data)
    #     logger.log('Got the Q values')

    # take the initial state:
    env.seed(test_env_seed)
    # analyse_q_vs_a_naive()
    analyse_q_tb()
    env.close()
