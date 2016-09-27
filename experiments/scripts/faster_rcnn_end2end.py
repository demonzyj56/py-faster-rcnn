#!/usr/bin/env python

# Translate `faster_rcnn_end2end.sh` using bash script to
# execute in python directly.

import datetime
import sys
import os
from timeit import default_timer as timer
# import caffe

class Logger(object):
    def __init__(self, log_file=None):
        self.terminal = sys.stdout
        if log_file is None:
            self.log = open("logfile.log", "a")
        else:
            self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def train(stps):
    import caffe
    from train_net import combined_roidb
    from fast_rcnn.train import train_net
    from fast_rcnn.config import cfg, get_output_dir
    print "Train with configs:"
    print cfg.TRAIN
    print "Setup:"
    print stps
    gpu_id = stps["GPU_ID"]
    solver = stps["TRAIN"]["SOLVER"]
    iters  = stps["TRAIN"]["ITERS"]
    imdb_name = stps["TRAIN"]["IMDB"]
    pretrained_model = stps["TRAIN"]["WEIGHTS"]
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    imdb, roidb = combined_roidb(imdb_name)
    output_dir = get_output_dir(imdb)
    print '{:d} roidb entries'.format(len(roidb))
    print 'Output will be saved to `{:s}`'.format(output_dir)
    train_net(solver, roidb, output_dir,
              pretrained_model=pretrained_model,
              max_iters=iters)




def faster_rcnn_end2end(config_file=None, setup_file=None, **kw):
    from fast_rcnn.config import cfg, cfg_from_file
    cfg_from_file(config_file)
    stps = _read_setup_file(setup_file)
    net = stps["NET"]
    dataset = stps["DATASET"]

    logfile = "experiments/logs/faster_rcnn_end2end_{}_{}.txt.".format(net, dataset) + \
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(logfile, 'a') as f:
        # redirect stdout to logfile
        sys.stdout = Logger(f)
        print "Logging output to {}".format(logfile)
        start = timer()
        train(stps)
        end = timer()
        print "elapse time: {} s".format(end-start)

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def _add_path():
    this_dir = os.path.dirname(__file__)
    lib_path = os.path.join(this_dir, '..', '..', 'lib')
    tools_path = os.path.join(this_dir, '..', '..', 'tools')
    add_path(lib_path)
    add_path(tools_path)
    import _init_paths

def _read_setup_file(setup_file):
    import yaml
    with open(setup_file, 'r') as f:
        return yaml.load(f)

if __name__ == '__main__':
    _add_path()
    faster_rcnn_end2end("experiments/cfgs/faster_rcnn_end2end.yml",
                        "experiments/cfgs/faster_rcnn_end2end_setup.yml")



