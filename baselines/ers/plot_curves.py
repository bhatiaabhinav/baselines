import argparse
import os
import time

import matplotlib.pyplot as plt

import numpy as np

from baselines.ers.args import str2bool

try:
    # Not necessary for monitor writing, but very useful for monitor loading
    import ujson as json
except ImportError:
    import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID',
                    default='BreakoutNoFrameskip-v4')
parser.add_argument(
    '--logdir', help='logs will be read from logdir/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR', default=os.getenv('OPENAI_LOGDIR'))
parser.add_argument(
    '--run_ids', help='run ids to plot. seperated by comma. Dont specify to plot all', default=None)
parser.add_argument(
    '--metrics', help='seperated by comma', default='Reward')
parser.add_argument('--smoothing', type=int, default=100)
parser.add_argument('--live', type=str2bool, default=False)
parser.add_argument('--style', default='seaborn')
parser.add_argument('--update_interval', type=int, default=30)
args = parser.parse_args()


def enter_figure(event):
    event.canvas.figure.autoscale = False
    # print("autoscale false")


def leave_figure(event):
    # global autoscale, time_mouse_leave
    event.canvas.figure.autoscale = True
    # time_mouse_leave = time.time()
    event.canvas.figure.time_mouse_leave = time.time()
    # print('leave_figure', event.canvas.figure)
    # event.canvas.figure.patch.set_facecolor('grey')
    # event.canvas.draw()
    # print("autoscale true")


def load_progress(fname, fields):
    progress_values = {}
    with open(fname, 'rt') as f:
        lines = f.readlines()
        for line in lines:
            log = json.loads(line)
            if log['Episode'] == 0 and len(progress_values) > 0:
                # break when episodes reset to zero. most probably due to end of training and start of testing
                break
            if log['Exploited']:
                for f in fields:
                    if f in progress_values.keys():
                        progress_values[f].append(log[f])
                    else:
                        progress_values[f] = []
    return progress_values


def moving_average(arr, n=30):
    if len(arr) == 0:
        return np.array([])
    cumsum = np.cumsum(np.insert(arr, 0, np.zeros(n)))
    if len(arr) < n:
        div = np.arange(1, len(arr) + 1)
    else:
        div = np.insert(n * np.ones(len(arr) - n + 1), 0, np.arange(1, n))
    return (cumsum[n:] - cumsum[:-n]) / div


def read_all_data(dirs, metrics):
    data = {}
    for dir_name in dirs:
        plot_name = dir_name
        if ':' in dir_name:
            plot_name = dir_name.split(':')[0]
            dir_name = dir_name.split(':')[1]
        logs_dir = os.path.join(args.logdir, args.env, dir_name)
        if os.path.isdir(logs_dir):
            fname = os.path.join(logs_dir, 'progress.json')
            if os.path.exists(fname):
                try:
                    plot_data = load_progress(fname, metrics)
                    data[dir_name] = {
                        "dir_name": dir_name,
                        "plot_name": plot_name,
                        "plot_data": plot_data
                    }
                except Exception as e:
                    print("Could not read {0}".format(dir_name))
                    print(type(e), e)
    return data


def plot_figure(data, metric):
    fig = plt.figure(num=metric)  # type: plt.Figure
    curves = []
    for dir_name, dir_data in data.items():
        try:
            x = np.array(dir_data['plot_data']['Episode'])
            y = np.array(dir_data['plot_data'][metric])
            y_av = moving_average(y, args.smoothing)
            curve, = plt.plot(x, y_av, label=dir_data['plot_name'])
            curves.append(curve)
        except Exception as e:
            print("Could not plot {metric} for {dir_name}".format(metric=metric, dir_name=dir_name))
            print(type(e), e)
    plt.xlabel('Episode no')
    plt.ylabel(metric)
    plt.legend()
    plt.title('Average {0} Per episode'.format(metric))
    if args.live:
        fig.canvas.mpl_connect('axes_enter_event', enter_figure)
        fig.canvas.mpl_connect('axes_leave_event', leave_figure)
        fig.autoscale = True
        fig.time_mouse_leave = time.time() - args.update_interval
    return fig, curves


def update_figure(fig: plt.Figure, curves, data, metric):
    plt.figure(num=fig.number)
    for dir_data, curve in zip(data.values(), curves):
        try:
            x = np.array(dir_data['plot_data']['Episode'])
            y = np.array(dir_data['plot_data'][metric])
            y_av = moving_average(y, args.smoothing)
            curve.set_xdata(x)
            curve.set_ydata(y_av)
        except Exception as e:
            print("Could not plot {metric} for {dir_name}".format(metric=metric, dir_name=dir_data['dir_name']))
            print(type(e), e)
    plt.gca().relim()
    plt.gca().autoscale(enable=plt.gcf().autoscale and time.time() -
                        plt.gcf().time_mouse_leave > args.update_interval)


def save_figure(fig, name):
    save_dir = os.path.join(args.logdir, args.env, "plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "{0}.png".format(name))
    fig.savefig(save_path)


if args.run_ids is None:
    dirs = os.listdir(os.path.join(args.logdir, args.env))
else:
    dirs = args.run_ids.split(',')

metrics = args.metrics.split(',')
metrics = [m.strip('\"') for m in metrics]

data = read_all_data(dirs, ["Episode"] + metrics)
last_update_at = time.time()
plt.style.use(args.style)

if args.live:
    plt.ion()

figs = []
curve_sets = []
for metric in metrics:
    fig, curves = plot_figure(data, metric)
    figs.append(fig)
    curve_sets.append(curves)
    save_figure(fig, metric)

while args.live:
    plt.pause(10)
    if time.time() - last_update_at >= args.update_interval:
        data = read_all_data(dirs, ["Episode"] + metrics)
        last_update_at = time.time()
        for fig, curves, metric in zip(figs, curve_sets, metrics):
            update_figure(fig, curves, data, metric)
            save_figure(fig, metric)
