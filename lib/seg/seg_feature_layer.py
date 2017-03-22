#!/usr/bin/evn python


from __future__ import print_function
import numpy as np
import caffe
import yaml
import cv2
import os.path as osp
from multiprocessing import Process, Queue, JoinableQueue
from fast_rcnn.config import cfg


class SegNetWrapper(Process):

    def __init__(self, in_queue, out_queue, gpu_id, net_def, net_weight):
        super(SegNetWrapper, self).__init__()
        self._net = None
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._gpu_id = gpu_id
        self._net_def = net_def
        self._net_weight = net_weight

    def run(self):
        ''' Forward the net and put output into queue.'''
        print("SegNetWrapper starts...")
        self.setup_caffe()
        while True:
            next_input = self._in_queue.get()
            if next_input is None:
                print('No input for SegNetWrapper, exiting...')
                #  self._in_queue.task_done()
                break

            # get data and forward
            data, rois = next_input
            self._net.blobs['data'].reshape(*data.shape)
            self._net.blobs['data'].data[...] = data
            self._net.blobs['rois'].reshape(*rois.shape)
            self._net.blobs['rois'].data[...] = rois
            self._net.forward()
            #  self._in_queue.task_done()
            self._out_queue.put(self._net.blobs['seg_features'].data)

        return

    def setup_caffe(self):
        # initial caffe again
        import caffe
        caffe.set_mode_gpu()
        caffe.set_device(self._gpu_id)
        self._net = caffe.Net(self._net_def, self._net_weight, caffe.TEST)


class SegFeatureLayer(caffe.Layer):

    def setup(self, bottom, top):
        try:
            self.layer_params = yaml.load(self.param_str_)
        except AttributeError:
            self.layer_params = yaml.load(self.param_str)
        except:
            raise
        if cfg.DEBUG:
            from ipdb import set_trace; set_trace()
        self.cfg_key = 'TRAIN' if self.phase == 0 else 'TEST'
        _gpu_id = self.layer_params.get('gpu_id', 0)
        _this_dir = osp.dirname(__file__)
        _net_def = osp.join(_this_dir, 'pascalcontext-fcn8s-deploy.prototxt')
        _net_weight = osp.join(cfg.DATA_DIR, 'pascalcontext-fcn8s',
                               'pascalcontext-fcn8s-heavy.caffemodel')
        self.seg_net = caffe.Net(_net_def, _net_weight, caffe.TEST)
        #  #  self.in_queue = JoinableQueue()
        #  self.in_queue = Queue(10)
        #  self.out_queue = Queue(10)
        #  self.seg_net = SegNetWrapper(self.in_queue, self.out_queue, _gpu_id, _net_def, _net_weight)
        #  self.seg_net.start()
        #
        #  # define cleanup function for SegNetWrapper
        #  def cleanup():
        #      print('Terminating SegNetWrapper...')
        #      #  self.in_queue.put(None)
        #      #  self.in_queue.join()
        #      self.seg_net.terminate()
        #      self.seg_net.join()
        #  import atexit
        #  atexit.register(cleanup)

        top[0].reshape(1, 60)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        #  self.in_queue.put((bottom[0].data, bottom[1].data))
        #  seg_features = self.out_queue.get()
        if cfg.DEBUG:
            from ipdb import set_trace; set_trace()
        self.seg_net.blobs['data'].reshape(*bottom[0].data.shape)
        self.seg_net.blobs['data'].data[...] = bottom[0].data
        self.seg_net.blobs['rois'].reshape(*bottom[1].data.shape)
        self.seg_net.blobs['rois'].data[...] = bottom[1].data
        seg_features = self.seg_net.forward()
        top[0].reshape(*seg_features['seg_features'].shape)
        top[0].data[...] = seg_features['seg_features']

    def backward(self, top, propagate_down, bottom):
        pass

