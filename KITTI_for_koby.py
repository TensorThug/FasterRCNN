# -*- coding: utf-8 -*-
# File: KITTI_for_koby.py

import numpy as np
import os
import tqdm
from PIL import Image
import json

from tensorpack.utils.timer import timed_operation
from NumpyEncoder_json import NumpyEncoder

valid_cat_names = ['Car', 'Person_sitting', 'Pedestrian']


class KITTIDetection(object):
    def __init__(self, basedir, name):
        self.name = name
        self._imgdir = os.path.realpath(os.path.join(basedir, 'images'))
        assert os.path.isdir(self._imgdir), self._imgdir
        self._labeldir = os.path.realpath(os.path.join(basedir, self.name))
        assert os.path.isdir(self._labeldir), self._labeldir

        included_extensions_labels = ['txt']
        included_extensions_imgs = ['png']
        self._label_list = [l_txt for l_txt in os.listdir(self._labeldir)
                            if(any(l_txt.endswith(ext) for ext in included_extensions_labels))]
        self._img_list = [img for img in os.listdir(self._imgdir)
                          if(any(img.endswith(ext) for ext in included_extensions_imgs))]
        assert len(self._label_list) == len(self._img_list)

        # background has class id of 0
        self.category_to_class_id = {'Car': 2, 'Person_sitting': 3, 'Pedestrian': 3}

    def load(self, add_gt=True):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts

        Returns:
            a list of dict, each has keys including:
                'height', 'width', 'file_name',
                and (if add_gt is True) 'boxes', 'class' and 'is_crowd'.
        """
        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):
            # list of dict
            imgs = []

            for Img in tqdm.tqdm(self._img_list):
                img_info = {}
                self._add_absolute_file_name_height_width(Img, img_info)
                if add_gt:
                    label = Img.split(".")[0] + '.txt'
                    assert os.path.isfile(os.path.join(self._labeldir, label)), os.path.join(self._labeldir, label)
                    self._add_detection_gt(label, img_info)
                imgs.append(img_info)
            return imgs

    def _add_absolute_file_name_height_width(self, img, img_info):
        """
        Add abosolute file name, height and width.
        """
        img_info['file_name'] = os.path.join(self._imgdir, img)
        assert os.path.isfile(img_info['file_name']), img_info['file_name']
        img_info['width'], img_info['height'] = Image.open(img_info['file_name']).size

    def _add_detection_gt(self, label, img_info):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        """
        boxes = []
        cls = []
        is_crowd = []

        with open(os.path.join(self._labeldir, label), 'r') as f:
            for line in f:
                fields = line.split(" ")
                if fields[0] in valid_cat_names:
                    cls.append(self.category_to_class_id[fields[0]])
                    [left, top, right, bottom] = np.float32(fields[4:8])
                    boxes.append([left, top, right, bottom])
                    is_crowd.append(False)

        if boxes:  # if "boxes" list isn't empty
            img_info['boxes'] = np.asarray(boxes, dtype='float32')  # nx4
            img_info['class'] = np.asarray(cls, dtype='int32')  # nx1
            img_info['is_crowd'] = np.asarray(is_crowd, dtype='int8')  # nx1

    @staticmethod
    def convert_to_desirable_format(dic):
        new_dic = {}
        if 'basedir' in dic.keys():
            new_dic = dic
        else:
            new_dic['file_name'] = dic['file_name']
            new_dic['height'] = int(dic['height'])
            new_dic['width'] = int(dic['width'])
            if 'boxes' in dic.keys():
                new_dic['boxes'] = np.asarray(dic['boxes'], dtype='float32')
                new_dic['class'] = np.asarray(dic['class'], dtype='int32')
                new_dic['is_crowd'] = np.asarray(dic['is_crowd'], dtype='int8')
        return new_dic

    @staticmethod
    def load_many(basedir, names, add_gt=True, _load_from_json_file=True):
        """
        Load and merges several instance files together.

        Returns the same format as :`KITTIDetection.load`.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]

        _load_flag = False
        if add_gt and os.path.exists(os.path.realpath(os.path.join(basedir, 'kitti_labels.json'))):
            dump_str = json.load(open(os.path.realpath(os.path.join(basedir, 'kitti_labels.json'))))
            dump = json.loads(dump_str, object_hook=KITTIDetection.convert_to_desirable_format)
            dic = dump[0]
            if (dic['basedir'] == basedir) and (len(dic['names']) == len(names)):
                _load_flag = True
                for name in names:
                    if not(name in dic['names']):
                        _load_flag = False
                        break

        ret = []
        if _load_from_json_file and _load_flag:
            ret = dump[1:]
        else:
            for n in names:
                kitti = KITTIDetection(basedir, n)
                ret.extend(kitti.load(add_gt))

            if add_gt:
                dic_initial = {'basedir': basedir, 'names': names}
                dumped_ret = [dic_initial]
                dumped_ret.extend(ret)
                dumped = json.dumps(dumped_ret, cls=NumpyEncoder)
                with open(os.path.join(os.path.realpath(basedir), 'kitti_labels.json'), 'w') as f:
                    json.dump(dumped, f)

        return ret
