#!/usr/bin/env python
"""
This module is used for loading pascal_voc under different scales.
By default the image set `JPEGImages` is replaced by the following folders:
JPEGImages_HR: Contains the original HR images.
JPEGImages_LR: Contains images that are downsampled and upsampled using bilinear interpolation.
JPEGImages_SR: Contains images that are downsampled and upsampled using super resolution method.
By default the dataset supports only testing on VOC2007.
"""
from datasets.pascal_voc import pascal_voc
from fast_rcnn.config import cfg
import os


class pascal_voc_sr(pascal_voc):
    def __init__(self, sr_kind, year, devkit_path=None):
        '''
        Here sr_kind is one of the following:
        HR: The original high resolution images.
        LR: HR images downsampled and then upsampled using bilinear interpolation.
        SR: HR images downsampled and then upsampled using super resolution method.
        '''
        self._sr_kind = sr_kind.upper()
        # reset to test setting
        super(pascal_voc_sr, self).__init__('test', year,
                                            devkit_path=devkit_path)

    def image_path_from_index(self, index):
        '''
        Assume that the images for the respective HR/LR/SR are located at
        JPEGImages_<HR/LR/SR> folder.
        '''
        image_path = os.path.join(self._data_path,
                                  'JPEGImages' + '_' + self._sr_kind,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'VOCdevkitSR' + self._year)


if __name__ == '__main__':
    pass
