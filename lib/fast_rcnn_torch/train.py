#!/usr/bin/env python

import os
from fast_rcnn.train import filter_roidb
from fast_rcnn.config import cfg
import imp

def get_model(solver_prototxt):
    d = os.path.dirname(solver_prototxt)
    name = '_'.join(os.path.dirname(p).split('/')[-3:])
    sw = imp.load_source(name, os.path.join(d, 'train.py'))
    return sw.TorchSolverWrapper

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    roidb = filter_roidb(roidb)
    sw = get_model(solver_prototxt)(
        solver_prototxt=solver_prototxt,
        roidb=roidb,
        output_dir=output_dir,
        gpu_id=cfg.GPU_ID,
        pretrained_model=pretrained_model
    )
    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
