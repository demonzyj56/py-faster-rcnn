#!/usr/bin/env python
# VGG16-Fast-RCNN network.

import torch
import torch.nn as nn
from fast_rcnn_torch.vgg16_base import get_vgg16_base
from fast_rcnn_torch.roi_pooling_layer import TorchROIPoolingLayer

class Net(nn.Module):
    def __init__(self, num_cls):
        super(Net, self).__init__()
        self._net_name = 'vgg16_fast_rcnn'
        self.conv_layers, self.fc_layers = get_vgg16_base()
        self.roi_pool_layer = TorchROIPoolingLayer(1./16, 7, 7)
        self.cls_score = nn.Linear(4096, num_cls)
        self.bbox_pred = nn.Linear(4096, num_cls*4)
        self._init_weights()

    def forward(self, data, rois):
        conv5_3 = self.conv_layers(data)
        pool5 = self.roi_pool_layer(conv5_3, rois)
        fc = self.fc_layers(pool5)
        return self.cls_score(fc), self.bbox_pred(fc)

    def _init_weights(self):
        # TODO: Set which layers should be backward and which should not.
        self.cls_score.weight.data.normal_(std=0.01)
        self.cls_score.bias.data.fill_(0)
        self.bbox_pred.weight.data.normal_(std=0.001)
        self.bbox_pred.bias.data.fill_(0)
