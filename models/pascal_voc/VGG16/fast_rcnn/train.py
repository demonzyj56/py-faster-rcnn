#!/usr/bin/env python

""" Training routine for this model.
    This file holds all `targets` and `optimizers` needed for
    this model.
"""

# TODO: write a base class so that this method could be reused.
# TODO: Add support for multigpu

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

from .model import Net


class TorchSolverWrapper(object):
    def __init__(self, solver_prototxt, roidb, output_dir, gpu_id,
                 pretrained_model=None):
        cuda = len(gpu_id) > 0
        self.net = Net(21)  # 20(fg) + 1(bg)
        self.criteria = OrderedDict((
            ('loss_cls', nn.CrossEntropyLoss()),
            ('loss_bbox', nn.SmoothL1Loss())
        ))
        self.data_loader = TorchRoiDataLoader(roidb, num_classes, cuda=cuda)
        if cuda:
            self.net.cuda()
        self.output_dir = output_dir
        self.gpu_id = gpu_id

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            self.bbox_means = torch.Tensor(self.bbox_means)
            self.bbox_stds = torch.Tensor(self.bbox_stds)
            print 'done'

        self._options_from_caffe_prototxt(solver_prototxt)
        self._cur_iter = 0
        self._step_iter = self.stepvalue.next()

        if pretrained_model is not None:
            self.net.load_state_dict(torch.load(pretrained_model))

    def step(self):
        train_blobs = self.data_loader.get()
        cls_score, bbox_pred = \
            self.net(train_blobs['data'], train_blobs['rois'])
        for v in self.criteria.values():
            v.zero_grad()
        loss_cls = self.criteria['loss_cls'](cls_score,
                                             train_blobs['labels'])
        loss_bbox = self.criteria['loss_bbox'](bbox_pred,
                                               train_blobs['bbox_targets'])
        loss_cls.backward()
        loss_bbox.backward()
        self.optim.step()
        self._cur_iter += 1
        if self._cur_iter == self._step_iter:
            self.lr *= self.gamma
            try:
                self._step_iter = self.stepvalue.next()
            except StopIteration:
                # -1 so that this step iter would be never reached.
                self._step_iter = -1

        return loss_cls, loss_bbox

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                             cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                             hasattr(self.net, 'bbox_pred'))

        if scale_bbox_params:
            orig_0 = self.net.bbox_pred.weight.data.clone()
            orig_1 = self.net.bbox_pred.bias.data.clone()

            net.bbox_pred.weight *= self.bbox_stds
            net.bbox_pred.bias = \
                net.bbox_pred.bias * self.bbox_stds + self.bbox_means

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.torchmodel')
        filename = os.path.join(self.output_dir, filename)

        torch.save(self.net.state_dict(), filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if scale_bbox_params:
            self.net.bbox_pred.weight.data = orig_0
            self.net.bbox_pred.bias.data = orig_1
        return filename

    def train_model(self, max_iters):
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        loss_cls_all, loss_bbox_all = [], []
        while self._cur_iter < max_iters:
            timer.tic()
            loss_cls, loss_bbox = self.step()
            timer.toc()
            loss_cls_all.append(loss_cls)
            loss_bbox_all.append(loss_bbox)

            if self._cur_iter % self.display == 0:
                print 'iter: {}'.format(self._cur_iter)
                print 'loss_cls: {:.3f}'.format(np.mean(loss_cls_all[-self.display:]))
                print 'loss_bbox: {:.3f}'.format(np.mean(loss_bbox_all[-self.display:]))

            if self._cur_iter % (10 * self.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self._cur_iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self._cur_iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

    @property
    def optim(self):
        if self._optim is None or self._cur_iter == self._step_iter:
            self._optim = self._get_optim()
        return self._optim

    @optim.setter
    def optim(self, out_val):
        """ Set optim manually. """
        self._optim = out_val

    def _get_optim(self):
        """ Construct optimizer on demand. """
        return torch.optim.SGD(self.net.parameters(),
            lr=self.base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay)

    def _options_from_caffe_prototxt(self, solver_prototxt):
        solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            try:
                pb2.text_format.Merge(f.read(), solver_param)
            except AttributeError:
                from google.protobuf import text_format
                text_format.Merge(f.read(), solver_param)

        assert solver_param.type == 'SGD', \
                'Support SGD solver only for now.'
        assert solver_param.lr_policy in \
                ['step', 'multistep']
        self.base_lr = solver_param.base_lr
        self.display = solver_param.display
        self.momentum = solver_param.momentum
        self.weight_decay = solver_param.weight_decay
        self.gamma = solver_param.weight_decay
        self.snapshot_prefix = solver_param.snapshot_prefix
        if len(self.stepvalue) > 0:
            assert solver_param.lr_policy == 'multistep'
            self.stepvalue = iter(solver_param.stepvalue)
        else:
            assert solver_param.lr_policy == 'step'
            self.stepvalue = itertools.count(
                start=solver_param.stepsize,
                step=solver_param.stepsize)
