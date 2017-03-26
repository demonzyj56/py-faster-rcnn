#!/usr/bin/env python
""" Torch implementation of ROI Pooling Layer. """

import torch
import torch.nn as nn
from torch.autograd import Variable


class TorchROIPoolingLayer(nn.Module):
    def __init__(self, spatial_scale, pooled_h, pooled_w):
        super(TorchROIPoolingLayer, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.rescale = nn.UpsamplingBilinear2d(size=(pooled_h, pooled_w))

    def forward(self, feat, rois):
        assert rois.size(1) == 5
        # pooled_feat = Variable(torch.zeros(rois.size(0), *feat.size()[1:]))
        inds = rois.data[:, 0].long()
        rois_data = rois.data[:, 1:] * self.feature_stride
        rois_data = rois_data.floor().long()
        pooled_feat = []
        for i in xrange(rois.size(0)):
            x1, y1, x2, y2 = rois_data[i]
            f = self.rescale(feat[inds[i], :, y1:(y2+1), x1:(x2+1)])
            pooled_feat.append(f)

        return torch.stack(pooled_feat, dim=0)


if __name__ == '__main__':
    roi_pool = TorchROIPoolingLayer(16, 7, 7)
    from IPython import embed; embed()
