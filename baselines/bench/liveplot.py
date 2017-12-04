from baselines import bench
import numpy as np
import matplotlib.pyplot as plt
import os
import time

try:
    import ujson as json # Not necessary for monitor writing, but very useful for monitor loading
except ImportError:
    import json

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
parser.add_argument('--logdir', help='logs will be read from logdir/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR', default=os.getenv('OPENAI_LOGDIR'))
parser.add_argument('--run_no', help='Run no', default=0)
args = parser.parse_args()

logs_dir = os.path.join(args.logdir, args.env, str(args.run_no))
fname = os.path.join(logs_dir, 'progress.json')

params_fname = os.path.join(logs_dir, 'params.json')
if os.path.exists(params_fname):
    print('Params:')
    with open(params_fname, 'r') as f:
        for l in f.readlines():
            print(l)

def load_progress(fname):
    logs = []
    with open(fname, 'rt') as f:
        lines = f.readlines()
        for line in lines:
            log = json.loads(line)
            if 'nupdates' in log: logs.append(log)
    return {
        'policy_entropy': [log['policy_entropy'] for log in logs],
        'explained_variance': [log['explained_variance'] for log in logs],
        'fps': [log['fps'] for log in logs],
        'timesteps': [log['total_timesteps'] for log in logs],
        'value_loss': [log['value_loss'] for log in logs],
        'nupdates': [log['nupdates'] for log in logs]
        }

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

x = [0]
y = [np.random.random()]

def enter_figure(event):
    event.canvas.figure.autoscale = False
    
    
def leave_figure(event):
    #global autoscale, time_mouse_leave
    event.canvas.figure.autoscale = True
    #time_mouse_leave = time.time()
    event.canvas.figure.time_mouse_leave = time.time()
    #print('leave_figure', event.canvas.figure)
    #event.canvas.figure.patch.set_facecolor('grey')
    #event.canvas.draw()

plt.figure(1)
rpe, = plt.plot(x,y, label='reward per episode')
av_rpe, = plt.plot(x,y, label='moving average(100)')
global_av_rpe, = plt.plot(x,y, label='overall average')
plt.xlabel('Episode no')
plt.ylabel('Reward')
plt.legend()
plt.title('Reward Per episode')
plt.gcf().canvas.mpl_connect('figure_enter_event', enter_figure)
plt.gcf().canvas.mpl_connect('figure_leave_event', leave_figure)
plt.gcf().autoscale = True
plt.gcf().time_mouse_leave = time.time() - 5

#-----------------------------------------------------

plt.figure(2)
lpe, = plt.plot(x,y, label='length per episode')
av_lpe, = plt.plot(x,y, label='moving average(100)')
global_av_lpe, = plt.plot(x,y, label='overall average')
plt.xlabel('Episode no')
plt.ylabel('Length in seconds')
plt.legend()
plt.title('Length Per episode')
plt.gcf().canvas.mpl_connect('figure_enter_event', enter_figure)
plt.gcf().canvas.mpl_connect('figure_leave_event', leave_figure)
plt.gcf().autoscale = True
plt.gcf().time_mouse_leave = time.time() - 5

#------------------------------------------------------

plt.figure(3)

plt.subplot(211)
entropy, = plt.plot(x,y)
plt.title('Policy entropy')
plt.xlabel('Timesteps')
plt.ylabel('Policy entropy for latest update')

plt.subplot(212)
explained_variance, = plt.plot(x,y)
plt.title('Explained variance')
plt.xlabel('Timesteps')
plt.ylabel('Explained variance for latest update')

plt.gcf().canvas.mpl_connect('figure_enter_event', enter_figure)
plt.gcf().canvas.mpl_connect('figure_leave_event', leave_figure)
plt.gcf().autoscale = True
plt.gcf().time_mouse_leave = time.time() - 5

#-----------------------------------------------------

plt.figure(4)

plt.subplot(211)
total_time, = plt.plot(x,y)
plt.title('Time')
plt.xlabel('Timesteps')
plt.ylabel('Time in mins')

plt.subplot(212)
fps, = plt.plot(x,y)
plt.title('FPS')
plt.xlabel('Timesteps')
plt.ylabel('FPS')

plt.gcf().canvas.mpl_connect('figure_enter_event', enter_figure)
plt.gcf().canvas.mpl_connect('figure_leave_event', leave_figure)
plt.gcf().autoscale = True
plt.gcf().time_mouse_leave = time.time() - 5


#-------------------------------------------------------

plt.figure(5)

plt.subplot(211)
nupdates, = plt.plot(x,y)
plt.title('Number of SGD updates')
plt.xlabel('Timesteps')
plt.ylabel('Count')

plt.subplot(212)
value_loss, = plt.plot(x,y)
plt.title('Value loss')
plt.xlabel('Timesteps')
plt.ylabel('Value loss')

plt.gcf().canvas.mpl_connect('figure_enter_event', enter_figure)
plt.gcf().canvas.mpl_connect('figure_leave_event', leave_figure)
plt.gcf().autoscale = True
plt.gcf().time_mouse_leave = time.time() - 5



while True:
    r = bench.load_results(logs_dir)

    plt.figure(1)
    y = r['episode_rewards']
    x = [i for i in range(len(y))]
    rpe.set_xdata(x)
    rpe.set_ydata(y)
    av_rpe.set_xdata(x)
    av_rpe.set_ydata(moving_average(y, 100))
    av = np.average(y) if len(y) > 0 else 0
    global_av_rpe.set_xdata(x)
    global_av_rpe.set_ydata([av for i in range(len(y))])
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time()-plt.gcf().time_mouse_leave>5)

    plt.figure(2)
    y = r['episode_lengths']
    x = [i for i in range(len(y))]
    lpe.set_xdata(x)
    lpe.set_ydata(y)
    av_lpe.set_xdata(x)
    av_lpe.set_ydata(moving_average(y, 100))
    av = np.average(y) if len(y) > 0 else 0
    global_av_lpe.set_xdata(x)
    global_av_lpe.set_ydata([av for i in range(len(y))])
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time()-plt.gcf().time_mouse_leave>5)

    r = load_progress(fname)


    plt.figure(3)

    plt.subplot(211)
    x = r['timesteps']
    y = r['policy_entropy']
    entropy.set_xdata(x)
    entropy.set_ydata(y)
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time()-plt.gcf().time_mouse_leave>5)

    plt.subplot(212)
    y = r['explained_variance']
    explained_variance.set_xdata(x)
    explained_variance.set_ydata(y)
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time()-plt.gcf().time_mouse_leave>5)


    plt.figure(4)

    plt.subplot(211)
    y = r['fps']
    total_time.set_xdata(x)
    total_time.set_ydata([float(x[i])/(60*y[i]) for i in range(len(x))])
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time()-plt.gcf().time_mouse_leave>5)

    plt.subplot(212)
    fps.set_xdata(x)
    fps.set_ydata(y)
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time()-plt.gcf().time_mouse_leave>5)

    
    plt.figure(5)

    plt.subplot(211)
    y = r['nupdates']
    nupdates.set_xdata(x)
    nupdates.set_ydata(y)
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time()-plt.gcf().time_mouse_leave>5)

    plt.subplot(212)
    y = r['value_loss']
    value_loss.set_xdata(x)
    value_loss.set_ydata(y)
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time()-plt.gcf().time_mouse_leave>5)
    

    
    plt.pause(5)