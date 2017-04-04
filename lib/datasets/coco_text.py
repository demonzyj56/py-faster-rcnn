from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import json
import uuid
from pycocotools.coco_text import COCO_Text
from utils.timer import Timer

# TODO: add legibility and language and class attribute filters.

class coco_text(imdb):
    """ Don't subclass coco.  Instead, create a thin wrapper
    around a coco object and call functions when needed.
    """
    def __init__(self, image_set):
        super(coco_text, self).__init__('coco_text_' + image_set)
        self._num_miniview = 1000
        self._overlap_thresh = 0.5
        self._view_map = {
            'minival': 'val'}
        self._data_path = osp.join(cfg.DATA_DIR, 'coco')
        self._data_name = 'train2014'  # all images are from coco train set
        self._image_set = image_set
        #  self._coco = coco('train', '2014')  # all images are from train set.
        self._ct = COCO_Text(self._get_ann_file())
        self._classes = ('__background__', '__foreground__')
        self._class_to_ind = {}
        self._class_to_coco_cat_id = {}
        self._image_index = self._load_image_set_index()
        self.set_proposal_method('gt')
        self.competition_mode(False)


    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        img = self._ct.loadImgs(index)[0]
        image_path = osp.join(self._data_path, 'images',
                              self._data_name, img['file_name'])
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_ann_file(self):
        return osp.join(self._data_path, 'annotations', 'COCO_Text.json')

    def _load_image_set_index(self):
        """
        Load image ids for coco text, which is a subset of respective data split.
        """
        if self._image_set in ('train', 'val', 'test'):
            return getattr(self._ct, self._image_set)  # train/val/test
        else:
            assert self._image_set in self._view_map.keys()
            image_set = self._view_map[self._image_set]
            index_set_index = getattr(self._ct, image_set)
            return index_set_index[:min(len(index_set_index), self._num_miniview)]

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_coco_text_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_coco_text_annotation(self, index):
        """
        coco returns dict with keys ('boxes', gt_classes',
        'gt_overlaps', 'flipped', 'seg_areas')
        """
        im_ann = self._ct.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        objs = self._ct.loadAnns(self._ct.getAnnIds(imgIds=index))
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.ones((num_objs), dtype=np.int32)  # always the same
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for ix, obj in enumerate(objs):
            boxes[ix, :] = obj['clean_bbox']
            seg_areas[ix] = obj['area']
            overlaps[ix, 1] = 1.0

        ds_utils.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def evaluate_detections(self, all_boxes, output_dir=None):
        if cfg.DEBUG:
            from ipdb import set_trace; set_trace()
        assert self._image_set != 'test', \
            'No gt annotation for test set in COCO Text.'
        gt_roidb = self.gt_roidb()
        num_pos = sum([r['boxes'].shape[0] for r in gt_roidb])

        all_inds = [np.ones((b.shape[0],), dtype=np.int)*i for i, b in \
            enumerate(all_boxes[1])]
        inds = np.concatenate(all_inds, axis=0).astype(np.int)
        boxes = np.concatenate(all_boxes[1], axis=0).astype(np.float32)
        scores = boxes[:, -1].copy()
        boxes = boxes[:, :-1].copy()

        num_dets = boxes.shape[0]
        tp = np.zeros((num_dets,), dtype=np.float32)
        fp = np.zeros((num_dets,), dtype=np.float32)


        sorted_inds = np.argsort(scores)[::-1]
        inds = inds[sorted_inds].ravel()
        boxes = boxes[sorted_inds, :]
        scores = scores[sorted_inds].ravel()

        assigned = [np.zeros((r['boxes'].shape[0], ), dtype=np.bool) \
            for r in gt_roidb]

        for d in xrange(num_dets):
            if gt_roidb[inds[d]]['boxes'].size == 0:
                fp[d] = 1
                continue
            ov = bbox_overlaps(
                np.ascontiguousarray(boxes[[d], :], dtype=np.float),
                np.ascontiguousarray(gt_roidb[inds[d]]['boxes'], dtype=np.float)
            ).ravel()
            max_ind = np.argmax(ov)
            if ov[max_ind] > self._overlap_thresh and \
                    not assigned[inds[d]][max_ind]:
                tp[d] = 1
                assigned[inds[d]][max_ind] = True
            else:
                fp[d] = 1

        fp = np.cumsum(fp).astype(np.float)
        tp = np.cumsum(tp).astype(np.float)
        rec = fp / num_pos
        prec = tp / (tp + fp)
        ap = 0.0
        for t in np.arange(0.0, 1.0+cfg.EPS, 0.01, dtype=np.float):
            pos = np.where(rec>t)[0]
            p = np.max(prec[pos]) if pos != [] else 0.0
            ap = ap + p / 101.0

        print '~~~~   Detection results for COCO Text {} ~~~~'.format(self._image_set)
        print '~~~~ Average Precision @ IoU 0.5: {:.3f} ~~~~'.format(ap)
        print '~~~~ Evaluate @ {} boxes per image on average ~~~~'.format(
            int(num_dets/len(gt_roidb)))

        if output_dir is not None:
            eval_file = osp.join(output_dir, 'detection_results.pkl')
            coco_eval = {'prec': prec, 'rec': rec, 'ap': ap}
            with open(eval_file, 'wb') as fid:
                cPickle.dump(coco_eval, fid, cPickle.HIGHEST_PROTOCOL)

