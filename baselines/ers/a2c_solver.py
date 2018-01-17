#!/usr/bin/env python
import os
import logging
import gym
import gym_ERSLE
import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, FcPolicy, ErsPolicy, ErsPolicy2, ErsPolicy3, FcWithBiasPolicy, BiasLcPolicy, RandomPolicy, NoOpPolicy
from baselines.common.atari_wrappers import ObsExpandWrapper


def train(env_id, ob_dtype, num_frames, seed, policy, lrschedule, ecschedule, num_cpu, nsteps, nstack, _lambda, saved_model_path, render, no_training):

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and
                                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)), allow_early_resets=(policy in ('greedy', 'ga', 'rat')))
            gym.logger.setLevel(logging.WARN)
            return ObsExpandWrapper(env)
            # return NoopFrameskipWrapper(ObsExpandWrapper(env))
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    ent_coef = 0.025
    env.id = env_id
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'fc':
        policy_fn = FcPolicy
        ent_coef = 0.1
    elif policy == 'fcwithbias':
        policy_fn = FcWithBiasPolicy
        ent_coef = 0.01
    elif policy == 'biaslc':
        policy_fn = BiasLcPolicy
        ent_coef = 0.06
    elif policy == 'ers':
        policy_fn = ErsPolicy
    elif policy == 'ers2':
        policy_fn = ErsPolicy2
        ent_coef = 0.03
    elif policy == 'ers3':
        policy_fn = ErsPolicy3
        ent_coef = 0.007
    elif policy == 'random':
        policy_fn = RandomPolicy
    elif policy == 'noop':
        policy_fn = NoOpPolicy

    learn(policy_fn, env, seed, ob_dtype=ob_dtype, total_timesteps=int(num_frames), frameskip=1, lrschedule=lrschedule, ecschedule=ecschedule, saved_model_path=saved_model_path, render=render, no_training=no_training,
          nsteps=nsteps, nstack=nstack, _lambda=_lambda, ent_coef=ent_coef)
    env.close()


def main():
    from baselines.ers.args import parse
    args = parse()
    train(args.env, ob_dtype=args.ob_dtype, num_frames=1e6 * args.million_frames, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule, ecschedule=args.ecschedule, num_cpu=args.num_cpu, nsteps=args.nsteps, nstack=args.nstack, _lambda=args._lambda,
          saved_model_path=args.saved_model, render=args.render, no_training=args.no_training)


if __name__ == '__main__':
    main()
