#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)

Train only the rpn part, and generate corresponding proposals.
"""

import _init_paths
import caffe
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil
from train_faster_rcnn_alt_opt import rpn_generate
from train_net import combined_roidb

def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--test_imdb', dest='test_imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--rpn_test', dest='rpn_test_prototxt',
                        help='rpn prototxt for testing',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train_rpn(queue=None, imdb_names=None, init_model=None, solver=None,
              max_iters=None, cfg=None):
    """Train a Region Proposal Network in a separate training process.
    """

    # Not using any proposals, just ground-truth boxes
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.BBOX_REG = False  # applies only to Fast R-CNN bbox regression
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    #  import caffe
    #  _init_caffe(cfg)

    #  roidb, imdb = get_roidb(imdb_name)
    imdb, roidb = combined_roidb(imdb_names)
    print 'roidb len: {}'.format(len(roidb))
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    rpn_model_path = model_paths[-1]
    # Send final model path through the multiprocessing queue
    #  queue.put({'model_path': rpn_model_path})
    return {'model_path': rpn_model_path}

def rpn_test(imdb_name=None, rpn_proposal_path=None, area='all'):
    """ Eval recall for generated proposals. """
    print 'RPN proposal: {}'.format(rpn_proposal_path)

    imdb = get_imdb(imdb_name)
    imdb.set_proposal_method('rpn')
    if rpn_proposal_path is not None:
        imdb.config['rpn_file'] = rpn_proposal_path


    ret = imdb.evaluate_recall(candidate_boxes=None, area=area)
    ar = ret['ar']
    gt_overlaps = ret['gt_overlaps']
    recalls = ret['recalls']
    thresholds = ret['thresholds']
    print 'Method: rpn'
    print 'Evaluate Area: {}'.format(area)
    print 'AverageRec: {:.3f}'.format(ar)

    def recall_at(t):
        ind = np.where(thresholds > t - 1e-5)[0][0]
        assert np.isclose(thresholds[ind], t)
        return recalls[ind]

    print 'Recall@0.5: {:.3f}'.format(recall_at(0.5))
    print 'Recall@0.6: {:.3f}'.format(recall_at(0.6))
    print 'Recall@0.7: {:.3f}'.format(recall_at(0.7))
    print 'Recall@0.8: {:.3f}'.format(recall_at(0.8))
    print 'Recall@0.9: {:.3f}'.format(recall_at(0.9))
    # print again for easy spreadsheet copying
    print '{:.3f}'.format(ar)
    print '{:.3f}'.format(recall_at(0.5))
    print '{:.3f}'.format(recall_at(0.6))
    print '{:.3f}'.format(recall_at(0.7))
    print '{:.3f}'.format(recall_at(0.8))
    print '{:.3f}'.format(recall_at(0.9))

def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

    # queue for communicated results between processes
    mp_queue = mp.Queue()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    rpn_stage1_out = train_rpn(imdb_names=args.imdb_name,
                               init_model=args.pretrained_model,
                               solver=args.solver,
                               max_iters=args.max_iters,
                               cfg=cfg)
    #  mp_kwargs = dict(
    #          queue=mp_queue,
    #          imdb_names=args.imdb_name,
    #          init_model=args.pretrained_model,
    #          solver=args.solver,
    #          max_iters=args.max_iters,
    #          cfg=cfg)
    #  p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    #  p.start()
    #  rpn_stage1_out = mp_queue.get()
    #  p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, generate proposals for the test set'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.test_imdb_name,
            rpn_model_path=str(rpn_stage1_out['model_path']),
            cfg=cfg,
            rpn_test_prototxt=args.rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out['proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    for area in ['all', 'small', 'medium', 'large']:

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Stage 1 RPN, eval recall with area {}'.format(area)
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

        rpn_test(imdb_name=args.test_imdb_name,
                 rpn_proposal_path=rpn_stage1_out['proposal_path'],
                 area=area)


if __name__ == '__main__':
    main()
