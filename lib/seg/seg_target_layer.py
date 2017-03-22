#!/usr/bin/evn python


import numpy as np
import caffe
import yaml
import cv2
from fast_rcnn.bbox_transform import clip_boxes
from fast_rcnn.config import cfg


class SegTargetLayer(caffe.Layer):
    '''
    This layer takes in the last layer output by fcn,
    and outputs the normalized count of proposed boxes as well as
    the the surrounding areas.

    Input:
    score [1x60xHxW]: the score map output by pascalcontext-fcn8s.
    rois [Nx5]: candidate bboxes from rpn.

    Output:
    seg_feat [Nx60]: normalized count for each part of the roi, where
    the roi is dilated by a factor gamma.
    '''

    def setup(self, bottom, top):
        try:
            self.layer_params = yaml.load(self.param_str_)
        except AttributeError:
            self.layer_params = yaml.load(self.param_str)
        except:
            raise
        self.cfg_key = 'TRAIN' if self.phase == 0 else 'TEST'
        self.gamma = self.layer_params.get('gamma', 1.8)
        top[0].reshape(1, 60)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        assert bottom[0].data.shape[0] == 1
        num_cls = bottom[0].data.shape[1]
        # create integral_map
        cls_map = np.argmax(bottom[0].data.squeeze(), axis=0)
        if cfg.DEBUG:
            from ipdb import set_trace; set_trace()

        # create dilated bounding boxes
        dilated_rois = self._compute_dilated_boxes(bottom[1].data[:, 1:], self.gamma)
        #  dilated_rois = np.round(bottom[1].data[:, 1:] * self.gamma)
        dilated_rois = clip_boxes(dilated_rois, bottom[0].data.shape[2:]).astype(np.int)

        counts = []
        for i in xrange(dilated_rois.shape[0]):
            counts.append(
                self._count_from_cls_map(cls_map, dilated_rois[i, :], num_cls))
        if cfg.DEBUG:
            print '[DEBUG] Forward in SegTargetLayer'
            from ipdb import set_trace; set_trace()
        counts = np.vstack(counts)
        if cfg.DEBUG:
            set_trace()
            areas = self._compute_area(dilated_rois)
        total_counts = np.sum(counts, axis=1)
        total_counts[total_counts == 0] = 1
        normalized_counts = counts.astype(np.float32) / total_counts[:, np.newaxis]
        top[0].reshape(normalized_counts.shape[0], num_cls)
        top[0].data[...] = normalized_counts.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass

    def _count_from_int_map(self, int_map, box):
        ''' Return integral_map from box. '''
        x1, y1, x2, y2 = box
        return int_map[:, y2+1, x2+1] - int_map[:, y2+1, x1] - \
            int_map[:, y1, x2+1] + int_map[:, y1, x1]

    def _count_from_cls_map(self, cls_map, box, num_cls):
        ''' Return histogram given the cls_map. '''
        x1, y1, x2, y2 = box
        return np.bincount(cls_map[y1:(y2+1), x1:(x2+1)].flatten(), minlength=num_cls)

    def _compute_area(self, boxes):
        ''' Compute areas of the boxes. '''
        return (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    def _compute_dilated_boxes(self, boxes, gamma):
        ''' Compute boxes with a dilate coefficient of gamma.
        The dilation is with respect to the CENTER of boxes.
        '''
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        dilated_boxes = np.zeros_like(boxes)
        dilated_boxes[:, 0] = ctr_x - 0.5 * widths * gamma
        dilated_boxes[:, 1] = ctr_y - 0.5 * heights * gamma
        dilated_boxes[:, 2] = ctr_x + 0.5 * widths * gamma
        dilated_boxes[:, 3] = ctr_y + 0.5 * heights * gamma
        return dilated_boxes



