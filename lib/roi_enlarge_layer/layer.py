import caffe
import numpy as np


def check(cond, err_msg=None):
    assert cond, err_msg


class RoiEnlargeLayer(caffe.Layer):
    ''' bottom[0] is the rois and bottom[1] is data.
        Using data to access image height and width.
        Otherwise it is unable to clip rois.
    '''

    def __getattr__(self, attr):
        ''' Since py-faster-rcnn use param_str_
        (I don't know why), transform here to
        `param_str`.
        '''
        if attr == 'param_str':
            return self.param_str_
        raise AttributeError()

    def setup(self, bottom, top):
        check(len(bottom) == 2)
        check(bottom[0].shape[1] == 5)
        param = eval(self.param_str)
        check('enlarge_factor' in param.keys())
        self._factor = param['enlarge_factor']
        self._width = bottom[1].shape[2]
        self._height = bottom[1].shape[3]

    def forward(self, bottom, top):
        top[0].data[...] = self._enlarge_rois(bottom[0])

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def _enlarge_rois(self, rois):
        ''' rois is of [ind, x1, y1, x2, y2] '''
        # ctr_x = (rois[:, 1] + rois[:, 3]) / 2.
        # ctr_y = (rois[:, 2] + rois[:, 4]) / 2.
        w = rois[:, 3] - rois[:, 1] + 1.0
        h = rois[:, 4] - rois[:, 2] + 1.0
        ctr_x = rois[:, 1] + 0.5 * w
        ctr_y = rois[:, 2] + 0.5 * h
        im_rois_enlarged = np.zeros_like(rois)
        im_rois_enlarged[:, 0] = rois[:, 0]
        im_rois_enlarged[:, 1] = ctr_x - w * self._factor * 0.5
        im_rois_enlarged[:, 2] = ctr_y - h * self._factor * 0.5
        im_rois_enlarged[:, 3] = ctr_x + w * self._factor * 0.5
        im_rois_enlarged[:, 4] = ctr_y + h * self._factor * 0.5
        return self._clip_rois(im_rois_enlarged)

    def _clip_rois(self, rois):
        rois[:, 1] = np.maximum(
            np.minimum(rois[:, 1], self._height - 1), 0)
        rois[:, 2] = np.maximum(
            np.minimum(rois[:, 2], self._width - 1), 0)
        rois[:, 3] = np.maximum(
            np.minimum(rois[:, 3], self._height - 1), 0)
        rois[:, 4] = np.maximum(
            np.minimum(rois[:, 4], self._width - 1), 0)
        return rois
