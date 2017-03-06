#!/usr/bin/evn python


import numpy as np
import caffe
import yaml
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import bbox_transform_inv
from fast_rcnn.config import cfg


class PythonFilterLayer(caffe.Layer):

    def setup(self, bottom, top):
        try:
            self.layer_params = yaml.load(self.param_str_)
        except AttributeError:
            self.layer_params = yaml.load(self.param_str)
        except:
            raise
        self.cfg_key = 'TRAIN' if self.phase == 0 else 'TEST'

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        raise NotImplementedError

    def backward(self, top, propagate_down, bottom):
        pass


class NmsLayer(PythonFilterLayer):
    '''
    Assume the input is N x 5, where each row is of
    [x1, y1, x2, y2, score].  Output a list of indices
    indicating which posisition to keep.
    If a second bottom blob is provided, the first blob is regarded
    as deltas whereas the second is the bbox to regress to.  The
    layer will perform `bbox_transform_inv` before nms.
    '''

    def setup(self, bottom, top):
        super(NmsLayer, self).setup(bottom, top)
        self.thresh = self.layer_params['thresh']

    def forward(self, bottom, top):
        # if not self.transformed:
        #     keep = nms(bottom[0].data, self.thresh)
        # else:
        #     raise NotImplementedError
        if len(bottom) == 1:
            keep = nms(bottom[0].data, self.thresh)
        else:
            pred_bbox = bbox_transform_inv(bottom[1].data, bottom[0].data[:, :4])
            keep = nms(np.vstack(pred_bbox, bottom[0].data[:, -1]), self.thresh)
        top[0].reshape(len(keep))
        top[0].data[...] = keep


class TopNLayer(PythonFilterLayer):
    '''
    Input is N x M, where one of the M columns is the scores
    for each sample.  Default is the last column.  Specify
    `score_column` to indicate which column should be regarded
    as scores if not the last one.
    If `reverse` is specified, take the bottom N targets instead
    of top N.
    '''

    def setup(self, bottom, top):
        super(TopNLayer, self).setup(bottom, top)
        self.top_N = self.layer_params.get('top_N', cfg[self.cfg_key].RPN_BATCHSIZE)
        self.score_column = int(self.layer_params.get('score_column', -1))
        self.reverse = self.layer_params.get('reverse', False)
        top[0].reshape(1)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top_N = min(self.top_N, bottom[0].shape[0])
        top[0].reshape(top_N)
        if len(bottom[0].shape) > 1:
            keep = np.argsort(bottom[0].data[:, self.score_column])
        else:
            keep = np.argsort(bottom[0].data)
        if not self.reverse:
            keep = keep[::-1]
        top[0].data[...] = keep[:top_N].squeeze()


class ThresholdLayer(PythonFilterLayer):
    '''
    Input is N x M, where one of the M columns is the socres
    that should apply thresholding for.  Default is the last column.
    Specify `score_column` to indicate which column should be regarded
    as scores if not the last one.
    If `reverse` is specified as Ture, take the targets that are below
    the threshold instead of over it.
    '''

    def setup(self, bottom, top):
        super(ThresholdLayer, self).setup(bottom, top)
        self.thresh = self.layer_params['thresh']
        self.score_column = int(self.layer_params.get('score_column', -1))
        self.reverse = self.layer_params.get('reverse', False)
        assert self.score_column < bottom[0].shape[1]

    def forward(self, bottom, top):
        if not self.reverse:
            keep = np.where(bottom[0].data[:, self.score_column] > self.thresh)[0]
        else:
            keep = np.where(bottom[0].data[:, self.score_column] <= self.thresh)[0]
        top[0].reshape(keep.shape[0])
        top[0].data[...] = keep
