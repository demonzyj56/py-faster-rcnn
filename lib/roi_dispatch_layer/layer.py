import numpy as np
import caffe
from fast_rcnn.config import cfg


class RoiDispatchLayer(caffe.Layer):

    def __getattr__(self, attr):
        ''' Since py-faster-rcnn use param_str_
        (I don't know why), transform here to
        `param_str`.
        '''
        if attr == 'param_str':
            return self.param_str_
        raise AttributeError()

    def setup(self, bottom, top):
        assert len(bottom) == 1
        assert len(top) == 4
        self._param = eval(self.param_str)
        assert 'lower_bound' in self._param.keys()
        assert 'upper_bound' in self._param.keys()
        # self._batch_size = cfg.TRAIN.BATCH_SIZE
        print '[DEBUG] bottom_size = {}'.format(list(bottom[0].shape))
        print '[DEBUG] batch_size = {:d}'.format(cfg.TRAIN.BATCH_SIZE)
        # assert bottom[0].shape[0] == self._batch_size
        self._small_rois, self._medium_rois, self._large_rois, self._arrangement = \
            self._dispatch_rois(bottom[0].data, self._param['lower_bound'], \
            self._param['upper_bound'])

    def reshape(self, bottom, top):
        top[0].reshape(*self._small_rois.shape)
        top[1].reshape(*self._medium_rois.shape)
        top[2].reshape(*self._large_rois.shape)
        top[3].reshape(self._batch_size)

    def forward(self, bottom, top):
        top[0].data[...] = self._small_rois
        top[1].data[...] = self._medium_rois
        top[2].data[...] = self._large_rois
        top[3].data[...] = self._arrangement

    def backward(self, top, propagate_down, bottom):
        pass

    def _dispatch_rois(self, rois, lb, ub):
        # Utility function that dispatches the rois into three parts:
        # 0 < rois_1 <= lower_bound^2;
        # lower_bound^2 < rois_2 <= upper_bound^2;
        # upper_bound^2 < rois_3.
        # rois is of shape Nx5, where the second dimension is arranged as
        # [n, x1, y1, x2, y2]
        # The last returns the rearrange of rois
        # print '[DEBUG] {}'.format(type(rois.data))
        areas = [self._roi_area(rois[i, :]) for i in xrange(rois.shape[0])]
        idx1, idx2, idx3 = [], [], []
        for i in xrange(len(areas)):
            if areas[i] <= lb ** 2:
                idx1.append(i)
            elif areas[i] <= ub ** 2:
                idx2.append(i)
            else:
                idx3.append(i)
        return rois[idx1, :], rois[idx2, :], rois[idx3, :], \
            self._index_reverse(idx1 + idx2 + idx3)

    def _roi_area(self, roi):
        # Utility function that computes roi area.
        # Roi is of shape [n, x1, y1, x2, y2]
        return (roi[3] - roi[1] + 1) * (roi[4] - roi[2] + 1)

    def _index_reverse(self, indices):
        ''' Reverse a vector of indices
        Suppose A[idx] = B, we want idx', such that
        B[idx'] = A.
        '''
        rev = [0] * len(indices)
        for i in xrange(len(indices)): 
            rev[indices[i]] = i
        return rev