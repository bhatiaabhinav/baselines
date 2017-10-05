#!/usr/bin/env python
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_frames, seed, policy, lrschedule, num_cpu, saved_model_path, render, no_training):
    num_timesteps = int(num_frames / 4 * 1.1) 
    # divide by 4 due to frameskip, then do a little extras so episodes end
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and 
                os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env.id = env_id
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, ob_dtype='uint8', total_timesteps=num_timesteps, frameskip=4, lrschedule=lrschedule, saved_model_path=saved_model_path, render=render, no_training=no_training)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6). '
        'This number gets divided by 4 due to frameskip', type=int, default=40)
    parser.add_argument('--logdir', help='logs will be saved to {logdir}/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR. run_no gets incremented automatically based on existance of previous runs in {logdir}/{env}/ . No logging if logdir is not provided and the env variable is not set', default=os.getenv('OPENAI_LOGDIR'))
    parser.add_argument('--saved_model', help='file from which to restore model. This file will not get overwritten when new model is saved. New models are always saved to {logdir}/{env}/{run_no}/model', default = None)
    parser.add_argument('--render', help='whether or not to render the env. False by default', default=False)
    parser.add_argument('--no_training', help='whether to just play without training', default=False)
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
    train(args.env, num_frames=1e6 * args.million_frames, seed=args.seed, 
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=16, saved_model_path=args.saved_model, 
        render=args.render, no_training=args.no_training)

if __name__ == '__main__':
    main()
