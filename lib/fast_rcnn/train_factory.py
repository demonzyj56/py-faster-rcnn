#!/usr/bin/env python
''' This file is a factory wrapper for alternating between caffe implementation
of fast_rcnn and torch implementation.
'''

from fast_rcnn.config import cfg

__func = None

if cfg.USE_TORCH_IMPL:
    def __train_impl(**kwargs):
        from fast_rcnn_torch.train import train_net
        return train_net(**kwargs)
    __func = lambda: __train_impl
else:
    from fast_rcnn.train import train_net
    __func = lambda: train_net

def get_train_func():
    return __func()
