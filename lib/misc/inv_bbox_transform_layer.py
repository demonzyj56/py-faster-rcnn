import numpy as np
import caffe


class InvBboxTransformLayer(caffe.Layer):
    '''
    This is a pure Python implementation of inverse
    bbox transform into caffe layer.
    This layers takes in bbox_delta proposed by the network
    and bbox to regress to and returns the refined bounding boxes.
    '''