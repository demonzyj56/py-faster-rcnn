#!/usr/bin/env python

""" Training routine for this model.
    This file holds all `targets` and `optimizers` needed for
    this model.
"""

# TODO: write a base class so that this method could be reused.

import torch
import torch.nn as nn
import torch.optim as optim
import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os
import pprint
from datasets.factory import get_imdb
from fast_rcnn_torch.roi_data_generator import TorchRoiDataLoader
from collections import OrderedDict

from caffe.proto import caffe_pb2
import google.protobuf as pb2


# class SolverWrapper(object):
#     """ Solver around everything needed for training.
#     Everything needed for subsequent function is `step`.
#     """
#
#     def __init__(output_dir, num_classes, cuda, pretrained_model=None):
#         pass
#
#     def step(self):
#         train_blobs = self._data_loader.get()
#         cls_score, bbox_pred = \
#             self.net(train_blobs['data'], train_blobs['rois'])
#         for v in self.criteria.values():
#             v.zero_grad()
#         loss_cls = self.criteria['loss_cls'](cls_score,
#                                              train_blobs['labels'])
#         loss_bbox = self.criteria['loss_bbox'](bbox_pred,
#                                                train_blobs['bbox_targets'])
#         loss_cls.backward()
#         loss_bbox.backward()
#         self.optim.step()
#         return loss_cls, loss_bbox
#
#     @property
#     def net(self):
#         return self._net
#
#     @net.setter
#     def net(self, val):
#         self._net = val
#
#     @property
#     def criteria(self):
#         return self._criteria
#
#     @targets.setter
#     def criteria(self, cri):
#         self._criteria = cri
#
#     @property
#     def optimizer(self):
#         return self._optim
#
#     @optimizer.setter
#     def optimizer(self, optim):
#         self._optim = optim
#
#     @property
#     def data_loader(self):
#         return self.data_loader
#
#     @data_loader.setter
#     def data_loader(self, roidb, num_classes, cuda):
#         self._data_loader = TorchRoiDataLoader(roidb, num_classes, cuda=cuda)
