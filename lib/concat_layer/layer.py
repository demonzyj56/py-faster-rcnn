import numpy as np
import caffe


def check(cond, *args):
    if args is None:
        args = "Execption"
    if cond is not True:
        raise Exception(args)


class PythonConcatLayer(caffe.Layer):

    def _compute_top_shape(self, s0, s1):
        return (s1[0], s0[1] + s1[1], s1[2], s1[3])

    def setup(self, bottom, top):
        check(len(bottom) == 2)
        check(bottom[0].shape[0] == bottom[1].shape[0])
        check(bottom[0].shape[2] >= bottom[1].shape[2])
        check(bottom[0].shape[2] - bottom[1].shape[2] <= 1)
        check(bottom[0].shape[3] >= bottom[1].shape[3])
        check(bottom[0].shape[3] - bottom[1].shape[3] <= 1)
        print "Bottom 0 shape: {}".format(bottom[0].data.shape)
        print "Bottom 1 shape: {}".format(bottom[1].data.shape)
        print "Shape after concat: {}".format(
            self._compute_top_shape(bottom[0].shape, bottom[1].shape))

    def reshape(self, bottom, top):
        top[0].reshape(
            *self._compute_top_shape(bottom[0].shape, bottom[1].shape))

    def forward(self, bottom, top):
        bottom0_clip = \
            bottom[0].data[:, :, :bottom[1].shape[2], :bottom[1].shape[3]]
        top[0].data[...] = \
            np.concatenate((bottom0_clip, bottom[1].data), axis=1)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[:, :, :bottom[1].shape[2], :bottom[1].shape[3]] = \
                top[0].diff[:, :bottom[0].shape[2]]
        if propagate_down[1]:
            bottom[1].diff[...] = top[0].diff[:, bottom[0].shape[2]:]
