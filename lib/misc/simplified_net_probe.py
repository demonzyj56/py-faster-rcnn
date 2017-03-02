#!/usr/bin/env python

import numpy as np
import caffe

def __print_blob_contents(net, blob_names):
    import numpy as np
    np.set_printoptions(threshold=np.nan)
    for name in blob_names:
        print name + ': '
        print net.blobs[name].data


#  def __compute_nonzero_vs_all(net, blob_name):
#      data = net.blobs[blob_name].data.flatten()
#      count = 0
#      for x in data:
#          if int(x) != 0:
#              count = count + 1
#      print kkk


def __print_all_blob_shape(net):
    for name, blob in net.blobs.iteritems():
        print name + '\t' + str(blob.data.shape)


def simplified_net_probe(net):
    cls_data = net.blobs['gt_cls'].data.astype(np.int).flatten()
    cls_counts = [len(np.where(cls_data == i)[0]) for i in xrange(21)]
    print 'Class distribution: {}'.format(cls_counts)
    obj_data = net.blobs['gt_objectness'].data.astype(np.int).flatten()
    obj_true= np.where(obj_data == 1)[0]
    obj_false = np.where(obj_data == 0)[0]
    print 'Object: {}, Non-object: {}'.format(len(obj_true), len(obj_false))
    cls_data_final = net.blobs['gt_cls_final'].data.astype(np.int).flatten()
    cls_final_counts = [len(np.where(cls_data_final == i)[0]) for i in xrange(21)]
    print 'Final class distribution to backprop: {}'.format(cls_final_counts)
    obj_data_final = net.blobs['gt_objectness_final'].data.astype(np.int).flatten()
    obj_final_true = np.where(obj_data_final == 1)[0]
    obj_final_false = np.where(obj_data_final == 0)[0]
    print "Final object vs non object to backprop: Object: {}, Non-object: {}".format(len(obj_final_true), len(obj_final_false))
    print '{} of {} objects are backproped'.format(len(obj_final_true), len(obj_true))


