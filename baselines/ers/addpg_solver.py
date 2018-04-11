"""
dualing network style advantageous DDPG
"""
import os
import os.path
import time
from typing import List  # noqa: F401

import gym
import numpy as np
import tensorflow as tf

import gym_ERSLE  # noqa: F401
from baselines import logger
from baselines.common.atari_wrappers import (BreakoutContinuousActionWrapper,
                                             ClipRewardEnv, EpisodicLifeEnv,
                                             FireResetEnv, FrameStack, MaxEnv,
                                             NoopResetEnv, SkipAndFrameStack,
                                             WarpFrame)
from baselines.ers.addpg_ac_model import Actor
from baselines.ers.analysis import analyse_q
from baselines.ers.experience_buffer import Experience, ExperienceBuffer
from baselines.ers.noise import NormalNoise  # noqa: F401
from baselines.ers.noise import OrnsteinUhlenbeckActionNoise
from baselines.ers.utils import normalize
from baselines.ers.wrappers import (CartPoleWrapper, ERSEnvImWrapper,
                                    ERSEnvWrapper)


def test_actor_on_env(sess, env_id, wrappers, learning=False, actor=None, seed=0, learning_env_seed=0,
                      test_env_seed=42, learning_episodes=40000, test_episodes=100, exploration_episodes=10, train_every=4,
                      mb_size=64, use_safe_noise=False, replay_memory_size_in_bytes=2 * 2**30, use_param_noise=False, init_scale=1e-3,
                      Noise_type=OrnsteinUhlenbeckActionNoise, exploration_sigma=0.2, exploration_theta=1, pre_training_steps=0,
                      gamma=0.99, double_Q_learning=False, advantage_learning=False, hard_update_target=False, tau=0.001,
                      render=False, render_mode='human', render_fps=60, save_path=None, load_path=None, **kwargs):
    np.random.seed(seed)
    env = gym.make(env_id)  # type: gym.Env
    for W in wrappers:
        env = W(env)  # type: gym.Wrapper
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
    if learning:
        experience_buffer = ExperienceBuffer(
            size_in_bytes=replay_memory_size_in_bytes)
        if use_param_noise:
            adaptive_sigma = init_scale
        else:
            noise = Noise_type(mu=np.zeros(env.action_space.shape),
                               sigma=exploration_sigma, theta=exploration_theta)

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
                count=mb_size))  # type: List[Experience]
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
        if not use_param_noise or no_explore:
            a, value, adv, q, A_summary, Q_summary = actor.get_a_V_A_Q([obs])
            a, value, adv, q = a[0], value[0], adv[0], q[0]
        if not no_explore:
            if use_param_noise:
                a = actor.get_noisy_a([obs])[0]
            else:
                a += noise()
                if 'ERS' in env_id:
                    a = normalize(a)
                else:
                    a = env.action_space.high * np.clip(a, -1, 1)
        if not use_param_noise or no_explore:
            logger.log('ep_f: {0}\tA: {1}\tQ: {2}'.format(
                ep_l, adv, q), level=logger.DEBUG)
            if f % 100 == 0:
                actor.writer.add_summary(A_summary, f)
                actor.writer.add_summary(Q_summary, f)
        return a

    Rs, no_explore_Rs, no_explore_blip_Rs, f = [], [], [], 0
    env.seed(learning_env_seed if learning else test_env_seed)
    pre_train = True
    for ep in range(learning_episodes if learning else test_episodes):
        obs, d, R, blip_R, ep_l = env.reset(), False, 0, 0, 0
        no_explore = (ep % 2 == 0) or not learning
        if not no_explore:
            if use_param_noise:
                if len(experience_buffer) >= mb_size:
                    mb_states = experience_buffer.random_states(mb_size)
                    if use_safe_noise:
                        actor.update_noisy_actor(
                            param_noise=actor.generate_safe_noise(adaptive_sigma, mb_states))
                    else:
                        actor.update_noisy_actor(
                            param_noise=actor.generate_normal_param_noise(adaptive_sigma))
                    divergence = actor.get_divergence_noisy_actor(mb_states)
                    logger.logkv('Exploitation Divergence', divergence)
                    actor.writer.add_summary(actor.divergence_summary.eval(
                        feed_dict={actor.divergence_placeholder: divergence}), ep)
                    multiplier = 1 + abs(divergence - exploration_sigma)
                    if divergence < exploration_sigma:
                        adaptive_sigma = adaptive_sigma * multiplier
                    else:
                        adaptive_sigma = adaptive_sigma / multiplier
                else:
                    adaptive_sigma = init_scale
                    actor.update_noisy_actor(
                        param_noise=actor.generate_normal_param_noise(adaptive_sigma))

                logger.logkv('Adaptive Sigma', adaptive_sigma)
                actor.writer.add_summary(actor.adaptive_sigma_summary.eval(
                    feed_dict={actor.adaptive_sigma_placeholder: adaptive_sigma}), ep)

            else:
                noise.reset()
        while not d:
            if learning and ep >= exploration_episodes and f % train_every == 0:
                train(pre_train=pre_train)
                pre_train = False
            a = act(obs)
            obs_, r, d, _ = env.step(a)
            if render:
                env.render(mode=render_mode)
                if render_fps is not None:
                    time.sleep(1 / render_fps)
            if learning:
                experience_buffer.add(Experience(obs, a, r, d, _, obs_))
            obs, R, f, ep_l = obs_, R + r, f + 1, ep_l + 1
            if 'blip_reward' in _:
                blip_R += _['blip_reward']
        Rs.append(R)
        if no_explore:
            no_explore_Rs.append(R)
            no_explore_blip_Rs.append(blip_R)
        logger.logkvs({
            'Episode': ep,
            'Reward': R,
            'Exploited': no_explore,
            'Blip_Reward': blip_R,
            'Length': ep_l,
            'Average Reward': np.average(Rs[-100:]),
            'Exploit Average Reward': np.average(no_explore_Rs[-100:]),
            'Exploit Average Blip Reward': np.average(no_explore_blip_Rs[-100:])
        })
        logger.dump_tabular()
        score_summary = actor.score_summary.eval(
            feed_dict={actor.R_placeholder: Rs[-1], actor.R_exploit_placeholder: no_explore_Rs[-1],
                       actor.blip_R_exploit_placeholder: no_explore_blip_Rs[-1]})
        actor.writer.add_summary(score_summary, ep)
        if save_path and ep % 50 == 0:
            actor.save(save_path)
            logger.log('model saved')
    env.close()
    logger.log('Average reward per episode: {0}'.format(np.average(Rs)))
    logger.log('Exploitation average reward per episode: {0}'.format(
        np.average(no_explore_Rs)))
    return actor


def main(seed=0, learning_env_seed=0, test_env_seed=42, test_mode=False, analysis_mode=False, save_path=None, load_path=None, **kwargs):
    with tf.Session() as sess:
        if not (test_mode or analysis_mode):
            logger.log('Training actor. seed={0}. learning_env_seed={1}'.format(
                seed, learning_env_seed))
            actor = test_actor_on_env(
                sess, learning=True, actor=None, **kwargs)
            actor.save(save_path)
            logger.log(
                'Testing actor. test_env_seed={0}'.format(test_env_seed))
            test_actor_on_env(sess, learning=False, actor=actor,
                              **dict(kwargs, save_path=None, load_path=None))
            logger.log('Testing done. Seeds were seed={0}. learning_env_seed={1}. test_env_seed={2}'.format(
                seed, learning_env_seed, test_env_seed))
            logger.log('Analysing model')
            analyse_q(sess, actor=actor, **dict(kwargs, load_path=None))
            logger.log("Analysis done. Results saved to logdir.")
        if test_mode:
            logger.log(
                'Testing actor. test_env_seed={0}'.format(test_env_seed))
            assert load_path is not None, "Please provide a saved model"
            test_actor_on_env(sess, learning=False, **
                              dict(kwargs, save_path=None))
            logger.log('Testing done. Seeds were seed={0}. learning_env_seed={1}. test_env_seed={2}'.format(
                seed, learning_env_seed, test_env_seed))
        if analysis_mode:
            logger.log('Analysing model')
            assert load_path is not None, "Please provide a saved model"
            analyse_q(sess, actor=None, **kwargs)
            logger.log("Analysis done. Results saved to logdir.")

        logger.log('-------------------------------------------------\n')


if __name__ == '__main__':
    # config = tf.ConfigProto(device_count={'GPU': 0})
    from baselines.ers.args import parse
    args = parse()
    logger.log('env_id: ' + args.env)
    logger.log('Seed: {0}'.format(args.seed))
    np.random.seed(args.seed)
    kwargs = vars(args).copy()
    kwargs['env_id'] = args.env
    kwargs['wrappers'] = []
    kwargs['replay_memory_size_in_bytes'] = args.replay_memory_gigabytes * 2**30
    kwargs['Noise_type'] = OrnsteinUhlenbeckActionNoise
    kwargs['learning_env_seed'] = args.seed
    kwargs['learning_episodes'] = args.training_episodes
    kwargs['test_env_seed'] = args.test_seed
    assert not (
        kwargs['use_layer_norm'] and kwargs['use_batch_norm']), "Cannot use both layer norm and batch norm"
    kwargs['save_path'] = os.path.join(logger.get_dir(), "model")
    kwargs['load_path'] = args.saved_model
    ERSEnvWrapper.k = args.nstack
    FrameStack.k = args.nstack
    if 'ERSEnv-ca' in kwargs['env_id']:
        kwargs['wrappers'] = [ERSEnvWrapper]
    elif 'ERSEnv-im' in kwargs['env_id']:
        kwargs['wrappers'] = [ERSEnvImWrapper]
    elif 'Pole' in kwargs['env_id']:
        kwargs['wrappers'] = [CartPoleWrapper]
    elif 'NoFrameskip' in kwargs['env_id']:
        kwargs['wrappers'] = [EpisodicLifeEnv, NoopResetEnv, MaxEnv, FireResetEnv,
                              WarpFrame, SkipAndFrameStack, ClipRewardEnv, BreakoutContinuousActionWrapper]
    main(**kwargs)
