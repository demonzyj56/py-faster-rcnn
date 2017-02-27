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

class GroundTruthLayer(caffe.layer):
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
        top[1]: bbox_deltas for each roi
        top[2]: cls for each roi (under a specific overlap value)
        """
        # parse the layer parameter string, which must be valid YAML
        try:
            layer_params = yaml.load(self.param_str_)
        except AttributeError:
            layer_params = yaml.load(self.param_str)
        except:
            raise
        self.obj_ov = layer_params['overlap']
        self.cls_ov = layer_params.get('cls_overlap', self.obj_ov)
        assert len(bottom) == 2
        assert bottom[0].shape[1] == 5
        assert bottom[1].shape[1] == 5

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        top_blobs = {'gt_objectness': None, 'gt_delta': None, 'gt_cls': None}
        top_names_to_inds = {'gt_objectness': 0, 'gt_delta': 1, 'gt_cls': 2}
        ov = bbox_overlaps(bottom[0].data[1:, :], bottom[1].data[:-1, :])

        # compute objectness
        max_ov = np.max(ov, axis=1)
        argmax_ov = np.argmax(ov, axis=1)
        obj_keep = np.where(max_ov > self.obj_ov)[0]
        objectness = np.zeros((bottom[0].shape[0], ), dtype=np.float)
        objectness[obj_keep] = 1.
        top_blobs['gt_objectness'] = objectness

        # compute class_label
        cls_keep = np.where(max_ov > self.cls_ov)[0]
        cls_labels = np.zeros((bottom[0].shape[0],))
        for k in cls_keep:
            cls_labels[k] = bottom[1].data[argmax_ov[k], -1]
        top_blobs['gt_cls'] = cls_labels

        # compute gt_delta
        gt_boxes_cord = bottom[1].data[argmax_ov, :-1]
        gt_delta = bbox_transform(bottom[0].data[:, 1:], gt_boxes_cord)
        top_blobs['gt_delta'] = gt_delta

        for name, blob in top_blobs.iteritems():
            ind = top_names_to_inds[name]
            top[ind].reshape(*(blob.shape))
            top[ind].data[...] = blob.astype(np.float32, copy=False)


    def backward(self, top, propogate_down, bottom):
        pass
