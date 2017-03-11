#!/usr/bin/env python


from __future__ import division
import numpy as np
import caffe
import cv2
import os.path as osp
from fast_rcnn.config import cfg


def fsrcnn(net, img, scale):
    '''Here img is raw image read by cv2.imread,
    namely, that img.dtype=np.uint8, and the color
    channel is BGR.
    return the same type and color channel as input image
    '''
    h, w, c = img.shape
    assert(c == 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_y, img_cr, img_cb = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    img_y = np.pad(img_y, ((0, 1), (0, 1)), mode='reflect')
    img_y = img_y.astype(np.float32)/255
    net.blobs['data'].reshape(1, 1, img_y.shape[0], img_y.shape[1])
    net.blobs['data'].data[...] = img_y
    net.forward()
    img_y_h = net.blobs['conv3'].data.squeeze()[:-1, :-1]
    assert(img_y_h.shape[0] == scale * h)
    assert(img_y_h.shape[1] == scale * w)
    img_y_h *= 255
    img_y_h[img_y_h <= 0] = 0
    img_y_h[img_y_h >= 255] = 255
    img_y_h = img_y_h.astype(np.uint8)
    img_cr_h = cv2.resize(img_cr, None, None, scale, scale)
    img_cb_h = cv2.resize(img_cb, None, None, scale, scale)
    img_h = np.stack((img_y_h, img_cr_h, img_cb_h), axis=-1)
    img_h = cv2.cvtColor(img_h, cv2.COLOR_YCrCb2BGR)
    return img_h


class SrNetHolder(object):
    def __init__(self):
        self._this_dir = osp.dirname(__file__)
        self._net_def = lambda td, s: osp.join(td, 'FSRCNN_deploy_x{}.prototxt').format(s)
        self._net_weight = lambda td, s: osp.join(td, 'x{}.caffemodel').format(s)
        self._nets = dict(zip((2, 3, 4), (None, None, None)))

    def get_net(self, scale):
        assert(scale in (2, 3, 4))
        if self._nets[scale] is None:
            self._nets[scale] = caffe.Net(self._net_def(self._this_dir, scale),
                                          self._net_weight(self._this_dir, scale),
                                          caffe.TEST)
        return self._nets[scale]

    def im_upscale(self, im, scale):
        if cfg.TRAIN.USE_SR:
            return fsrcnn(self.get_net(scale), im, scale)
        else:
            raise NotImplementedError

