#!/usr/bin/env python
from __future__ import division
import numpy as np
import caffe
import yaml
from utils.timer import Timer
from fast_rcnn.config import cfg


class TransposeLayer(caffe.Layer):
    """
    This layer reshape and transpose the feature map into
    the same order by proposal layer.
    For proposal_layer, it transposes a blob from (1, KxA, H, W) to
    (1, H, W, KxA) and reshape to (HxWxA, K) where rows are ordered
    by (H, W, A).
    """
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            layer_params = yaml.load(self.param_str)
        except:
            raise
        assert len(bottom) == 1
        assert bottom[0].shape[0] == 1
        self.num_anchors = layer_params['num_anchors']
        self.feat_lens = bottom[0].channels // self.num_anchors
        feat_lens = bottom[0].channels // self.num_anchors
        num_samples = bottom[0].height * bottom[0].width * self.num_anchors
        top[0].reshape(num_samples, feat_lens)
        self.ftimer, self.btimer = Timer(), Timer()

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        self.ftimer.tic()
        feat_lens = bottom[0].channels // self.num_anchors
        num_samples = bottom[0].height * bottom[0].width * self.num_anchors
        top[0].reshape(num_samples, feat_lens)
        top[0].data[...] = \
            bottom[0].data.transpose((0, 2, 3, 1)).reshape((-1, self.feat_lens))
        if cfg.DEBUG:
            print '[{}] top shape: {}'.format(type(self).__name__, str(top[0].data.shape))
            print '[{}] Forward time: {:3f}s'.format(type(self).__name__, self.ftimer.toc(average=False))
            from ipdb import set_trace; set_trace()


    def backward(self, top, propogate_down, bottom):
        bottom[0].diff[...] = \
            top[0].diff.reshape(
                (1, bottom[0].height, bottom[0].width, -1)
            ).transpose((0, 3, 1, 2))
