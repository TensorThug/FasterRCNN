# -*- coding: utf-8 -*-
# File: open_images_for_koby.py

import numpy as np
import os
import tqdm
from PIL import Image
import pandas as pd
import json

from tensorpack.utils.timer import timed_operation
from NumpyEncoder_json import NumpyEncoder


class OpenImagesDetection(object):
    def __init__(self, basedir, name):
        self.name = name
        self._imgdir = os.path.realpath(os.path.join(basedir, self.name))
        assert os.path.isdir(self._imgdir), self._imgdir
        annotation_file = os.path.join(
            basedir, 'boxes/{}-annotations-bbox.csv'.format(self.name))
        assert os.path.isfile(annotation_file), annotation_file
        classes_file = os.path.join(basedir, 'class-descriptions-boxable.csv')
        assert os.path.isfile(classes_file), classes_file

        # Load csv:
        self.classes = pd.read_csv(classes_file, names=['Label', 'Class'])
        self.boxes = pd.read_csv(annotation_file)

        # background has class id of 0
        self.category_to_class_id = {self.getMID('Tank'): 1, self.getMID('Car'): 2, self.getMID('Person'): 3}
        self.valid_MID = [self.getMID('Tank'), self.getMID('Car'), self.getMID('Person')]

        # remove lines where the field 'IsGroupOf' equals 1
        new_indices = self.boxes['IsGroupOf'] != 1
        list_indices = self.boxes.index[new_indices].tolist()  # No. of rows the new_indices belong to
        self.boxes = self.boxes[new_indices]

        # remove lines with irrelevant classes
        relevant_indices = np.full(len(self.boxes), False).tolist()
        for MID in self.valid_MID:
            tmp_indices = (self.boxes['LabelName'] == MID).tolist()
            relevant_indices = [relevant_indices[i] or tmp_indices[i] for i in range(len(relevant_indices))]
        ind = pd.Series(relevant_indices, index=list_indices)
        self.boxes = self.boxes[ind]

        # extract image list from the 'boxes' DataFrame
        self._img_list = list(np.unique(np.asarray(self.boxes['ImageID'].tolist())))

    def getMID(self, class_name):
        # Get MID of specific class
        MID = self.classes.loc[self.classes['Class'] == class_name]['Label'].tolist()[0]
        return MID

    def load(self, add_gt=True):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts

        Returns:
            a list of dict, each has keys including:
                'height', 'width', 'file_name', 'id'
                and (if add_gt is True) 'boxes', 'class' and 'is_crowd'.
        """
        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):
            # list of dict
            imgs = []

            for Img in tqdm.tqdm(self._img_list):
                img_info = {}
                img_info['id'] = Img
                img_path = os.path.join(self._imgdir, Img + '.jpg')
                assert os.path.isfile(img_path), img_path
                self._add_absolute_file_name_height_width(Img + '.jpg', img_info)
                if add_gt:
                    self._add_detection_gt(Img, img_info)
                imgs.append(img_info)
            return imgs

    def load_img_annotations(self, imID, add_gt=True):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dict

        Returns:
            a list of dict (1 dict) containing the following keys:
                'height', 'width', 'id', 'file_name',
                and (if add_gt is True) 'boxes', 'class' and 'is_crowd'.
        """
        img_path = os.path.join(self._imgdir, imID + '.jpg')
        if os.path.exists(img_path):
            img_info = {}
            img_info['id'] = imID
            assert os.path.isfile(img_path), img_path
            self._add_absolute_file_name_height_width(imID + '.jpg', img_info)
            if add_gt:
                self._add_detection_gt(imID, img_info)
            img_info = [img_info]
        else:
            img_info = []

        return img_info

    def _add_absolute_file_name_height_width(self, img, img_info):
        """
        Add abosolute file name, height and width.
        """
        img_info['file_name'] = os.path.join(self._imgdir, img)
        assert os.path.isfile(img_info['file_name']), img_info['file_name']
        img_info['width'], img_info['height'] = Image.open(img_info['file_name']).size

    def _add_detection_gt(self, imageID, img_info):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        """
        boxes = []
        cls = []
        is_crowd = []

        # pick the lines with the relevant image ID
        img_boxes = self.boxes.loc[self.boxes['ImageID'] == imageID]
        # remove lines where the field 'IsGroupOf' equals 1
        img_boxes = img_boxes.loc[img_boxes['IsGroupOf'] != 1]

        for index, row in img_boxes.iterrows():
            if row['LabelName'] in self.valid_MID:  # if the target is one of the valid targets
                cls.append(self.category_to_class_id[row['LabelName']])
                left = np.multiply(row['XMin'], img_info['width'] - 1, dtype='float32')
                right = np.multiply(row['XMax'], img_info['width'] - 1, dtype='float32')
                top = np.multiply(row['YMin'], img_info['height'] - 1, dtype='float32')
                bottom = np.multiply(row['YMax'], img_info['height'] - 1, dtype='float32')
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
            new_dic['id'] = dic['id']
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

        Returns the same format as :`OpenImagesDetection.load`.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]

        _load_flag = False
        if add_gt and os.path.exists(os.path.realpath(os.path.join(basedir, 'open_images_labels.json'))):
            dump_str = json.load(open(os.path.realpath(os.path.join(basedir, 'open_images_labels.json'))))
            dump = json.loads(dump_str, object_hook=OpenImagesDetection.convert_to_desirable_format)
            dic = dump[0]
            if (dic['basedir'] == basedir) and (len(dic['names']) == len(names)):
                _load_flag = True
                for name in names:
                    if not (name in dic['names']):
                        _load_flag = False
                        break

        ret = []
        if _load_from_json_file and _load_flag:
            ret = dump[1:]
        else:
            for n in names:
                open_images = OpenImagesDetection(basedir, n)
                ret.extend(open_images.load(add_gt))

            if add_gt:
                dic_initial = {'basedir': basedir, 'names': names}
                dumped_ret = [dic_initial]
                dumped_ret.extend(ret)
                dumped = json.dumps(dumped_ret, cls=NumpyEncoder)
                with open(os.path.join(os.path.realpath(basedir), 'open_images_labels.json'), 'w') as f:
                    json.dump(dumped, f)

        return ret

    @staticmethod
    def load_img_from_many(dict_open_images_annt, imID, add_gt=True):
        """
        Load and merges several instance files together.

        Returns the same format as :`OpenImagesDetection.load_img_annotations`.
        """
        ret = []
        for key in dict_open_images_annt:
            ret.extend(dict_open_images_annt[key].load_img_annotations(imID, add_gt))
        return ret

    @staticmethod
    def load_annt(basedir, names):
        """
        Returns the relevant classes after the annotation were loaded
        """
        if not isinstance(names, (list, tuple)):
            names = [names]
        dict_open_images_annt = {}
        for n in names:
            dict_open_images_annt[n] = OpenImagesDetection(basedir, n)
        return dict_open_images_annt
