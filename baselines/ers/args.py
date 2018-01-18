import ast
import os
import os.path
from baselines import logger


def parse():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='pyERSEnv-ca-dynamic-1440-v4')
    parser.add_argument(
        '--ob_dtype', help='datatype of observations eg. uint8, float32', default='float32')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=[
                        'cnn', 'lstm', 'lnlstm', 'fc', 'ers', 'ers2', 'ers3', 'fcwithbias', 'biaslc', 'random', 'noop', 'greedy', 'ga', 'rat'], default='fc')
    parser.add_argument('--lrschedule', help='Learning rate schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--ecschedule', help='Entropy coefficient schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--million_frames', help='How many frames to train (/ 1e6)',
                        type=float, default=7.2)  # 5000 = 625 * 8 episodes
    # parser.add_argument('--million_frames', help='How many frames to train (/ 1e6)', type=float, default=0.71424) # 520 = 65 * 8 episodes
    parser.add_argument('--num_cpu', help='Number of parallel environments', type=int, default=16)
    parser.add_argument(
        '--nsteps', help='an update happens every nsteps timesteps for each env', type=int, default=5)
    parser.add_argument(
        '--nstack', help='how many frames to stack to create one obs', type=int, default=1)
    parser.add_argument('--_lambda', help='lambda=1 => use nsteps returns. lambda=0 => use 1 step returns. intermidiate values cause averaging of various step returns. Equivalent to eligibility traces', type=float, default=0.95)
    parser.add_argument(
        '--logdir', help='logs will be saved to {logdir}/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR. run_no gets incremented automatically based on existance of previous runs in {logdir}/{env}/ . No logging if logdir is not provided and the env variable is not set', default=os.getenv('OPENAI_LOGDIR'))
    parser.add_argument(
        '--saved_model', help='file from which to restore model. This file will not get overwritten when new model is saved. New models are always saved to {logdir}/{env}/{run_no}/model', default=None)
    parser.add_argument(
        '--render', help='whether or not to render the env. False by default', type=bool, default=False)
    parser.add_argument(
        '--no_training', help='whether to just play without training', type=bool, default=False)
    parser.add_argument('--run_no_prefix', default='run')
    parser.add_argument('--replay_memory_gigabytes', type=float, default=2.0)
    parser.add_argument('--init_scale', type=float, default=0.0001)
    parser.add_argument('--nn_size', default="[400,300,300]")
    parser.add_argument('--generations', type=int, default=1000)
    parser.add_argument('--population_size', type=int, default=30)
    parser.add_argument('--truncation_size', type=int, default=7)
    parser.add_argument('--mutation_sigma', type=float, default=0.03)

    args = parser.parse_args()

    args.nn_size = ast.literal_eval(args.nn_size)

    if args.logdir:
        for run_no in range(int(1e6)):
            logdir = os.path.join(args.logdir, args.env, args.run_no_prefix +
                                  '_' + str(run_no).zfill(3))
            if not os.path.isdir(logdir):
                os.putenv('OPENAI_LOGDIR', logdir)
                logger.reset()
                logger.configure(logdir)
                break
            else:
                run_no += 1

    return args
