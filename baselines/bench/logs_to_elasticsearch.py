from baselines import bench
import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow
import sys
import os
import ntpath
import time
from glob import glob
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
from threading import Thread
from datetime import datetime

try:
    import ujson as json # Not necessary for monitor writing, but very useful for monitor loading
except ImportError:
    import json

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
parser.add_argument('--logdir', help='logs will be read from logdir/{env}/{run_no}/  . Defaults to os env variable OPENAI_LOGDIR', default=os.getenv('OPENAI_LOGDIR'))
parser.add_argument('--run_no', help='Run no', default=0)
parser.add_argument('--index_frames', default=False)

args = parser.parse_args()

index_name = str.lower('rl_logs_' + args.env + '_' + str(args.run_no))
logs_dir = os.path.join(args.logdir, args.env, str(args.run_no))
progress_fname = os.path.join(logs_dir, 'progress.json')
params_fname = os.path.join(logs_dir, 'params.json')
def get_monitor_files(dir, extn):
    return glob(os.path.join(dir, "*" + extn))
episodes_fnames = get_monitor_files(logs_dir, 'monitor.json')
frames_fnames = get_monitor_files(logs_dir, 'monitor_frames.json')
class LoadMonitorResultsError(Exception):
    pass
if not episodes_fnames:
    raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % ('monitor.json', logs_dir))

def watch(filename, indefinitely=True):
    with open(filename, 'r') as fp:
        hasNewLines = False
        while True:
            new = fp.readline()

            if new:
                if not hasNewLines:
                    print('Found new lines in ' + ntpath.basename(filename))
                hasNewLines = True
                l = new.strip()
                if len(l) > 0:
                    yield(l)
            else:
                if indefinitely:
                    if hasNewLines:
                        print('Waiting for more lines in ' + ntpath.basename(filename))
                        hasNewLines = False
                        yield None # yield none to signal temporary eof to go ahead with bulk upload
                    hasNewLines = False
                    time.sleep(5)
                else:
                    break

def find_global_tstart():
    ans = sys.float_info.max
    files = episodes_fnames + [progress_fname] if os.path.exists(progress_fname) else episodes_fnames
    for f in files:
        with open(f, 'r') as fp:
            line1 = fp.readline()
            obj = json.loads(line1)
            if 't_start' in obj:
                tstart = obj['t_start']
                if tstart < ans:
                    ans = tstart
    return ans

es = Elasticsearch()

global_tstart = find_global_tstart()

# first put params file in:
if os.path.exists(params_fname):
    for l in watch(params_fname):
        if l:
            obj = json.loads(l)
            obj['env_id'] = str(args.env)
            obj['run_no'] = str(args.run_no)
            obj['abstime'] = datetime.utcfromtimestamp(obj['abstime'])
            es.index(index=index_name, doc_type='params', id='params', body=obj)
            break


class FileWatchThread(Thread):
    def __init__(self, filename, doc_type, env_rank, **kwargs):
        super().__init__()
        self.filename = filename
        self.doc_type = doc_type
        self.env_rank = env_rank
        self.tstart = None
        self.episode_no = 0 # for backward compatability i.e. for episode logs which did not have episode_no, we must add here.
        self.kwargs = kwargs

    def process_log(self, line):
        line = line.replace('NaN', '0.0')
        obj = json.loads(line)
        if 't_start' in obj: #its the header line
            self.tstart = obj['t_start']
            return None
        else: # must be a log line
            obj['env_id'] = str(args.env)
            obj['run_no'] = str(args.run_no)
            if self.env_rank: obj['env_rank'] = self.env_rank
            if self.tstart and 't' in obj:
                obj['abstime'] = datetime.utcfromtimestamp(self.tstart + obj['t'])
                obj['t'] = obj['t'] + self.tstart - global_tstart
            action = {}
            if self.doc_type == 'frame':
                action['_id'] = str(self.env_rank) + '_' + str(obj['frame_no'])
            elif self.doc_type == 'episode':
                if not 'episode_no' in obj:
                    obj['episode_no'] = self.episode_no
                    self.episode_no += 1
                action['_id'] = str(self.env_rank) + '_' + str(obj['episode_no'])
            elif self.doc_type == 'progress':
                action['_id'] = str(obj['nupdates'])
            else:
                action['_id'] = ''
            action['_id'] = '_'.join([self.doc_type, action['_id']])
            action['_type'] = self.doc_type
            action['_index'] = index_name
            action['_source'] = obj
            return action

    def actions_stream(self):
        for line in watch(self.filename):
            if line:
                action = self.process_log(line)
                if action:
                    yield action
            else:
                yield None # to signal to index the batch so far

    def run(self):
        batch = []
        for a in self.actions_stream():
            if a: batch.append(a)
            if (len(batch) > self.kwargs.get('chunk_size', 500) or not a) and len(batch) > 0:
                bulk(es, batch, stats_only=True, **self.kwargs)
                batch.clear()


threads = []

if os.path.exists(progress_fname):
    threads.append(FileWatchThread(progress_fname, 'progress', None))

for ef in episodes_fnames:
    threads.append(FileWatchThread(ef, 'episode', int(''.join(filter(str.isdigit, ntpath.basename(ef))))))

if args.index_frames and frames_fnames and len(frames_fnames) > 0:
    for ff in frames_fnames:
        threads.append(FileWatchThread(ff, 'frame', int(''.join(filter(str.isdigit, ntpath.basename(ff))))))

for t in threads:
    t.setDaemon(True)
    t.start()

while True:
    time.sleep(1000)