from baselines import bench
import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow
import os
import time

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
parser.add_argument('--logdir', help='logs will be read from logdir/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR', default=os.getenv('OPENAI_LOGDIR'))
parser.add_argument('--run_no', help='Run no', default=0)
args = parser.parse_args()

logs_dir = os.path.join(args.logdir, args.env, str(args.run_no))


def moving_average(arr, n = 30):
    if len(arr) == 0:
        return np.array([])
    cumsum = np.cumsum(np.insert(arr, 0, np.zeros(n)))
    if len(arr) < n:
        div = np.arange(1, len(arr) + 1)
    else:
        div = np.insert(n * np.ones(len(arr) - n + 1), 0, np.arange(1, n))
    return (cumsum[n:] - cumsum[:-n]) / div

plt.ion()

autoscale = True
time_mouse_leave = time.time()-5

x = [0]
y = [np.random.random()]

def enter_figure(event):
    global autoscale
    autoscale = False
    
def leave_figure(event):
    global autoscale, time_mouse_leave
    autoscale = True
    time_mouse_leave = time.time()
    #print('leave_figure', event.canvas.figure)
    #event.canvas.figure.patch.set_facecolor('grey')
    #event.canvas.draw()

rpe, = plt.plot(x,y, label='reward per episode')
av_rpe, = plt.plot(x,y, label='moving average(100)')
plt.legend()
plt.title('Reward Per episode')
plt.gcf().canvas.mpl_connect('figure_enter_event', enter_figure)
plt.gcf().canvas.mpl_connect('figure_leave_event', leave_figure)


while True:
    r = bench.load_results(logs_dir)
    x = [i for i in range(len(r['episode_rewards']))]
    y = r['episode_rewards']
    rpe.set_xdata(x)
    rpe.set_ydata(y)
    av_rpe.set_xdata(x)
    av_rpe.set_ydata(moving_average(y, 100))
    plt.gca().relim()
    plt.gca().autoscale(enable=autoscale and time.time()-time_mouse_leave>5)
    plt.pause(5)
