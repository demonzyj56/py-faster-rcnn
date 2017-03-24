#!/usr/bin/env python

""" This is a python implementation of cross entropy layer.
Assume that that the input given is after log-softmax.
This is to ease the computation of gradients:

loss(x, target) = 1/N \sum target_i x (log(target_i) - x_i)

This mimics the setting of KLDivLoss used in pytorch.
"""

import caffe
import numpy as np
import yaml

EPS = 1e-14

class PyCrossEntropyLayer(caffe.Layer):

    def setup(self, bottom, top):
        try:
            self.layer_params = yaml.load(self.param_str_)
        except AttributeError:
            self.layer_params = yaml.load(self.param_str)
        except:
            raise
        self.cfg_key = 'TRAIN' if self.phase == 0 else 'TEST'

        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top[0].data[...] = \
            np.vdot(bottom[1].data, np.log(bottom[1].data+EPS)-bottom[0].data) / bottom[0].data.shape[0]

    def backward(self, top, propagate_down, bottom):
        assert not propagate_down[1]
        bottom[0].diff[...] = -bottom[1].data / bottom[1].data.shape[0]

