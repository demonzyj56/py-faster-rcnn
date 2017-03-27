#!/usr/bin/env python

import torch
import torch.nn as nn
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
import itertools

from caffe.proto import caffe_pb2
import google.protobuf as pb2

#  class SolverWrapperBase(object):
#      """ A simple wrapper for all options needed for training.
#      Note that we parse Caffe's solver prototxt only through
#      protobuf.  No actual caffe solver is created.
#      """
#      def __init__(self, roidb, output_dir, num_classes, cuda,
#                   solver_prototxt, pretrained_model=None):
#          """Initialize the SolverWrapper."""
#          self.output_dir = output_dir
#          self._num_classes = num_classes
#
#          if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
#              cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
#              # RPN can only use precomputed normalization because there are no
#              # fixed statistics to compute a priori
#              assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED
#
#          if cfg.TRAIN.BBOX_REG:
#              print 'Computing bounding-box regression targets...'
#              self.bbox_means, self.bbox_stds = \
#                      rdl_roidb.add_bbox_regression_targets(roidb)
#              print 'done'
#
#          #  self.solver_param = caffe_pb2.SolverParameter()
#          #  with open(solver_prototxt, 'rt') as f:
#          #      try:
#          #          pb2.text_format.Merge(f.read(), self.solver_param)
#          #      except AttributeError:
#          #          from google.protobuf import text_format
#          #          text_format.Merge(f.read(), self.solver_param)
#          #
#          #  self._net = _get_torch_net_from_path(self.solver_param.train_net,
#          #                                       num_classes)
#          #  if cuda:
#          #      self._net.cuda()
#          #
#          #  if pretrained_model is not None:
#          #      print ('Loading pretrained model '
#          #             'weights from {:s}').format(pretrained_model)
#          #      # TODO: Load pretrained torch model
#          #
#          #  self._init_criteria()
#          #  self._init_optim()
#
#          self._data_loader = TorchRoiDataLoader(roidb, num_classes, cuda=cuda)
#
#          self._iter = 0
    #
    #  def train_model(self, max_iters):
    #      """ Network training loop. """
    #      self._net.train()
    #      last_snapshot_iter = -1
    #      timer = Timer()
    #      model_paths = []
    #      while self._iter < max_iters:
    #          timer.tic()
    #          self.step()
    #          timer.toc()
    #          if self.solver.iter % (10 * self.solver_param.display) == 0:
    #              print 'speed: {:.3f}s / iter'.format(timer.average_time)
    #          if self._iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
    #              last_snapshot_iter = self._iter
    #              model_paths.append(self.snapshot())
    #
    #      if last_snapshot_iter != self._iter:
    #          model_paths.append(self.snapshot())
    #
    #      return model_paths
    #
    #  def step(self):
    #      train_blobs = self._data_loader.get()
    #      cls_score, bbox_pred = \
    #          self._net(train_blobs['data'], train_blobs['rois'])
    #      for v in self._criteria.values():
    #          v.zero_grad()
    #      loss_cls = self._criteria['loss_cls'](cls_score,
    #                                            train_blobs['labels'])
    #      loss_bbox = self._criteria['loss_bbox'](bbox_pred,
    #                                              train_blobs['bbox_targets'])
    #      loss_cls.backward()
    #      loss_bbox.backward()
    #      self._optim.step()
    #      self._iter += 1
    #
    #      # TODO: Update optimizer at step
    #      # TODO: Print loss value
    #      # TODO: Make compartible with rpn and faster rcnn
    #      # TODO: Abstract further for the whole training process:
    #      #       The optim and criteria and net should be packed together.
    #
    #
    #  def snapshot(self):
    #      pass
    #
    #  def _init_optim(self):
    #      """ Initialize optimizer from solver prototxt for training. """
    #      assert self.solver_param.lr_policy == 'step'  # one step only
    #      self._base_lr = self.solver_param.base_lr
    #      self._momentum = self.solver_param.momentum
    #      self._weight_decay = self.solver_param.weight_decay
    #      self._gamma = self.solver_param.gamma
    #      self._stepsize = self.solver_param.stepsize[0]
    #      self._optim = optim.SGD(self._net.parameters(),
    #                              lr=base_lr,
    #                              momentum=momentum,
    #                              weight_decay=weight_decay)
    #      if self._cuda:
    #          self._optim.cuda()
    #
    #  def _init_criteria(self):
    #      """ This function returns all criteria (loss) in training.
    #      The number of criteria is exactly the same as number and order of
    #      returns in forward.
    #      """
    #      self._criteria = OrderedDict((
    #          ('loss_cls', nn.CrossEntropyLoss()),
    #          ('loss_bbox', nn.SmoothL1Loss())
    #      ))
    #
def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    raise NotImplementedError


class OptimizerBase(object):
    def __init__(self):
        self._optim = None


class OptimizerCaffe(OptimizerBase):
    def __init__(self, model_param, solver_prototxt):
        """ Initialize from a caffe solver prototxt.
        Note that a caffe object is not construct, only prototxt is
        parsed.  The fields that we concern are:
        base_lr, lr_policy, momentum, weight_decay, gamma, stepsize,
        stepvalue
        """
        super(OptimizerCaffe, self).__init__()
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
        if len(self.stepvalue) > 0:
            assert solver_param.lr_policy == 'multistep'
            self.stepvalue = iter(solver_param.stepvalue)
        else:
            assert solver_param.lr_policy == 'step'
            self.stepvalue = itertools.count(
                start=solver_param.stepsize,
                step=solver_param.stepsize)
        self._cur_iter = 0
        self._step_iter = self.stepvalue.next()


    def step(self):
        """ At step, do the following:
        Step the optimizer;
        Increase iter count;
        Check if the current iter is at the step value, if so,
        then update the lr and step value.
        """
        self.optim.step()
        self._cur_iter += 1
        if self._cur_iter == self._step_iter:
            self.lr *= self.gamma
            try:
                self._step_iter = self.stepvalue.next()
            except StopIteration:
                # -1 so that this step iter would be never reached.
                self._step_iter = -1

    def _get_optim(self):
        return torch.optim.SGD(self.model_param,
            lr=self.base_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay)

    @property
    def optim(self):
        if self._optim is None or self._cur_iter == self._step_iter:
            self._optim = self._get_optim()
        return self._optim

    @optim.setter
    def optim(self, out_val):
        """ Set optim manually. """
        self._optim = out_val

    @property
    def iter(self):
        return self._cur_iter
