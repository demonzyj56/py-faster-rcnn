#!/usr/bin/env python
# This is the base part for VGG16 network, that can be resued by
# subsequent networks.

import torch
import torch.nn as nn
import torchvision

def get_vgg16_base(pretrained=True, remove_last=True):
    vgg16 = torchvision.models.vgg16(pretrained=pretrained)
    if remove_last:
        delattr(vgg16.features, '30')  # remove last max pool
        delattr(vgg16.classifier, '6')  # remove last linear layer
    return vgg16.features, vgg16.classifier


if __name__ == '__main__':
    features, classifier = get_vgg16_base()
