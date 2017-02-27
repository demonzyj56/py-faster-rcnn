#!/usr/bin/env python
import numpy as np
import caffe
import yaml


class TopNWeightLayer(caffe.Layer):
    ''' This layer takes in a loss vector of size Nx1, and output a 
    vector of size Nx1, where the position of topN losses are 1, and 
    the rest are zero.
    This layer is combined with other loss layers to back prop only
    topN losses.
    '''

    def setup(self, bottom, top):
        try:
            self.layer_params = yaml.load(self.param_str_)
        except AttributeError:
            self.layer_params = yaml.load(self.param_str)
        except:
            raise
        assert len(bottom) == 1
        assert bottom[0].shape[1] == 1
        if self.layer_params['top_N'] > 0:
            self.top_N = min(self.layer_params['top_N'], bottom[0].shape[0])
        else:
            self.top_N = bottom[0].shape[0]
        self.reverse = self.layer_params.get('reverse', False)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        keep = np.argsort(bottom[0].data)
        if not self.reverse:
            keep = keep[::-1]
        weight = np.zeros((self.top_N, ), dtype=np.float32)
        weight[keep] = 1.
        top[0].reshape(self.top_N)
        top[0].data[...] = weight

    def backward(self, top, propagate_down, bottom):
        pass
