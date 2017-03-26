#!/usr/bin/env python

"""ROI data generator that serves as RoiDataLayer."""
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import torch
from torch.autograd import Variable


class TorchRoiDataLoader(object):
    """
    Generate data each time it is called. Note that
    the class holds an `roidb` instead of `imdb`, which means
    that the generator is ignorant of which dataset is used.
    """
    def __init__(self, roidb, num_classes, cuda=True):
        self._roidb = roidb
        self._num_classes = num_classes
        self._cuda = cuda
        self._shuffle_roidb_inds()

    def get(self):
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]

        return self._get_torch_minibatch(minibatch_db)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_torch_minibatch(self, minibatch_db):
        """ return a dict with the following field:
        RPN case:
            data
            gt_boxes
            im_info
        None-RPN case (Selective Search):
            data
            rois
            labels
            bbox_targets
            bbox_inside_weights
            bbox_outside_weights
        """
        blobs = get_minibatch(minibatch_db, self._num_classes)
        torch_blobs = {}
        for k, v in blobs.iteritems():
            torch_blobs[k] = Variable(torch.Tensor(v))
            if self._cuda:
                torch_blobs[k] = torch_blobs[k].cuda()

        return torch_blobs
