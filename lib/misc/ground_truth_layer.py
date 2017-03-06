#!/usr/bin/env python

import numpy as np
import caffe
import yaml
try:
    from utils.cython_bbox import bbox_overlaps
except ImportError:
    print "Using python version of bbox_overlaps"
    from utils.python_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
from fast_rcnn.config import cfg
from utils.timer import Timer


class GroundTruthLayer(caffe.Layer):
    """
    This is the layer that takes in ground truth info
    (labels and bboxes) and outputs suitable labels
    for training.
    """
    def setup(self, bottom, top):
        """
        bottom[0]: rois (im_batch + predicted_bboxes)
        bottom[1]: gt_boxes (boxes + cls_ind)
        output:
        top[0]: objectness for each roi (under a specific overlap value)
        top[1]: bbox_deltas (under a specific overlap value)
        top[2]: cls for each roi (under a specific overlap value)
        top[3]: bbox_inds for input rois to keep for bbox regression
        """
        # parse the layer parameter string, which must be valid YAML
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            layer_params = yaml.load(self.param_str)
        except:
            raise
        #  self.obj_ov = layer_params.get('overlap', cfg.TRAIN.FG_THRESH)
        #  self.cls_ov = layer_params.get('cls_overlap', self.obj_ov)
        #  self.bbox_ov = layer_params.get('bbox_overlap', cfg.TRAIN.BBOX_THRESH)
        self.bbox_ov = cfg.TRAIN.BBOX_THRESH
        assert len(bottom) == 2
        top[0].reshape(1)
        top[1].reshape(1, 4)
        top[2].reshape(1)
        top[3].reshape(1)
        self.ftimer = Timer()

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        self.ftimer.tic()
        top_blobs = {'gt_objectness': None, 'gt_delta': None, 'gt_cls': None, 'roi_inds': None}
        top_names_to_inds = {'gt_objectness': 0, 'gt_delta': 1, 'gt_cls': 2, 'roi_inds': 3}
        ov = bbox_overlaps(bottom[0].data[:, 1:].astype(np.float),
                           bottom[1].data[:, :-1].astype(np.float))
        max_ov = np.max(ov, axis=1)
        argmax_ov = np.argmax(ov, axis=1)
        pos = np.where(max_ov > self.bbox_ov)[0]

        if cfg.DEBUG:
            np.set_printoptions(threshold=np.nan)
            #  from ipdb import set_trace; set_trace()
            print 'Number of gt bbox: {}'.format(bottom[1].shape[0])
            print 'Number of bbox filtered: {}'.format(len(pos))

        # Objectness
        objectness = np.zeros((bottom[0].shape[0], ), dtype=np.float)
        objectness[pos] = 1.
        top_blobs['gt_objectness'] = objectness

        # cls
        cls_labels = np.zeros((bottom[0].shape[0], ), dtype=np.float)
        for i in pos:
            cls_labels[i] = bottom[1].data[argmax_ov[i], -1]
        top_blobs['gt_cls'] = cls_labels

        # BBOX
        pos = np.where(max_ov > 0.1)[0]
        if len(pos) == 0:
            pos = np.where(max_ov > 0)[0]
            assert len(pos) > 0
        if cfg.DEBUG:
            #  set_trace()
            print 'New number of bbox to backprop: {}'.format(len(pos))

        if len(pos) > 0:
            bbox_keep = bottom[0].data[pos, 1:]
            bbox_gt = [bottom[1].data[argmax_ov[i], :-1] for i in pos]
            bbox_gt = np.vstack(bbox_gt)
            gt_delta = bbox_transform(bbox_keep, bbox_gt)
            top_blobs['gt_delta'] = gt_delta
        else:
            top_blobs['gt_delta'] = np.zeros((0, 4), dtype=np.float)

        # BBOX inds
        if len(pos) > 0:
            top_blobs['roi_inds'] = pos
        else:
            top_blobs['roi_inds'] = np.zeros((0, ), dtype=np.float)

        for name, blob in top_blobs.iteritems():
            ind = top_names_to_inds[name]
            top[ind].reshape(*(blob.shape))
            top[ind].data[...] = blob.astype(np.float32, copy=False)

        if cfg.DEBUG:
            print '[{}] Forward time: {:3f}s'.format(type(self).__name__, self.ftimer.toc(average=False))
            from ipdb import set_trace; set_trace()

    def backward(self, top, propogate_down, bottom):
        pass
