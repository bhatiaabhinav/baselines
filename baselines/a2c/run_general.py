#!/usr/bin/env python
import os, logging, gym
import numpy as np
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, FcPolicy, ErsPolicy, ErsPolicy2, ErsPolicy3, FcWithBiasPolicy, RandomPolicy, NoOpPolicy
import gym_ERSLE

class ObsExpandWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        self.observation_space = gym.spaces.Box(low if np.isscalar(low) else np.asarray(low).item(0), high if np.isscalar(high) else np.asarray(high).item(0),
                                               shape = (
                                                   env.observation_space.shape[0],
                                                   env.observation_space.shape[1] if len(env.observation_space.shape) >= 2 else 1,
                                                   env.observation_space.shape[2] if len(env.observation_space.shape) >= 3 else 1
                                                   ))

    def reset(self):
        ob = super().reset()
        if ob.ndim == 1:
            ob = ob[:, np.newaxis, np.newaxis]
        elif ob.ndim == 2:
            ob = ob[:, :, np.newaxis]
        return ob

    def step(self, action):
        ob, r, d, _ = super().step(action)
        if ob.ndim == 1:
            ob = ob[:, np.newaxis, np.newaxis]
        elif ob.ndim == 2:
            ob = ob[:, :, np.newaxis]
        return ob, r, d, _

def train(env_id, ob_dtype, num_frames, seed, policy, lrschedule, num_cpu, nsteps, nstack, _lambda, saved_model_path, render, no_training):
    
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and 
                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return ObsExpandWrapper(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    ent_coef = 0.01
    env.id = env_id
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    elif policy == 'fc':
        policy_fn = FcPolicy
        ent_coef = 0.005
    elif policy == 'fcwithbias':
        policy_fn = FcWithBiasPolicy
        ent_coef = 0.015
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
    learn(policy_fn, env, seed, ob_dtype=ob_dtype, total_timesteps=int(num_frames), frameskip=1, lrschedule=lrschedule, saved_model_path=saved_model_path, render=render, no_training=no_training,
          nsteps=nsteps, nstack=nstack, _lambda=_lambda, ent_coef=ent_coef)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='ERSEnv-v2')
    parser.add_argument('--ob_dtype', help='datatype of observations eg. uint8, float32', default='float32')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'fc', 'ers', 'ers2', 'ers3', 'fcwithbias', 'random', 'noop'], default='fc')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6)', type=int, default=40)
    parser.add_argument('--num_cpu', help='Number of parallel environments', type=int, default=16)
    parser.add_argument('--nsteps', help='an update happens every nsteps timesteps for each env', type=int, default=5)
    parser.add_argument('--nstack', help='how many frames to stack to create one obs', type=int, default=1)
    parser.add_argument('--_lambda', help='lambda=1 => use nsteps returns. lambda=0 => use 1 step returns. intermidiate values cause averaging of various step returns. Equivalent to eligibility traces', type=float, default=0.8)
    parser.add_argument('--logdir', help='logs will be saved to {logdir}/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR. run_no gets incremented automatically based on existance of previous runs in {logdir}/{env}/ . No logging if logdir is not provided and the env variable is not set', default=os.getenv('OPENAI_LOGDIR'))
    parser.add_argument('--saved_model', help='file from which to restore model. This file will not get overwritten when new model is saved. New models are always saved to {logdir}/{env}/{run_no}/model', default = None)
    parser.add_argument('--render', help='whether or not to render the env. False by default', type=bool, default=False)
    parser.add_argument('--no_training', help='whether to just play without training', type=bool, default=False)
    args = parser.parse_args()
    if args.logdir:
        for run_no in range(int(1e6)):
            logdir = os.path.join(args.logdir, args.env, str(run_no))
            if not os.path.isdir(logdir):
                os.putenv('OPENAI_LOGDIR', logdir)
                logger.reset()
                logger.configure(logdir)
                break
            else:
                run_no += 1
    train(args.env, ob_dtype=args.ob_dtype, num_frames=1e6 * args.million_frames, seed=args.seed, 
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=args.num_cpu, nsteps=args.nsteps, nstack=args.nstack, _lambda=args._lambda,
        saved_model_path=args.saved_model, render=args.render, no_training=args.no_training)

if __name__ == '__main__':
    main()
