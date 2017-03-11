# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
import pprint
from fast_rcnn.config import cfg
from fsrcnn import FSRCNN

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def prep_im_for_blob_sr(im, pixel_means, target_size, max_size):
    '''This function would upscale the image using
    super resolution method instead of bilinear
    upascaling.  Then the image would be downscaled
    to desired size.
    '''
    im_size_min, im_size_max = np.min(im.shape[:2]), np.max(im.shape[:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    if im_scale < 1:
        scale = 1
        im_scale_actual = im_scale
    elif im_scale < 2:
        scale = 2
        im_scale_actual = im_scale / 2.
    elif im_scale < 3:
        scale = 3
        im_scale_actual = im_scale / 3.
    else:
        scale = 4
        im_scale_actual = im_scale / 4.
    if scale > 1:
        im = FSRCNN(im, scale)
    im = cv2.resize(im, None, None, fx=im_scale_actual,
                    fy=im_scale_actual, interpolation=cv2.INTER_LINEAR)
    assert(np.min(im.shape[:2]) == target_size or
           np.max(im.shape[:2]) == max_size)
    im = im.astype(np.float32, copy=False)
    im -= pixel_means

    return im, im_scale


def im_to_lr(im, decimate):
    #  im = cv2.GaussianBlur(im, (5, 5), 3)
    im = cv2.resize(im, None, None, 1./decimate, 1./decimate)
    im = cv2.resize(im, im.shape[:2])
    return im

