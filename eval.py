# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
import numpy as np
import cv2
import time
from shapely.geometry import box as box_shapely
import matplotlib.pyplot as plt

from tensorpack.utils.utils import get_tqdm_kwargs

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as cocomask

from common import CustomResize, clip_boxes
from config import config as cfg

DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


def fill_full_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret


def detect_one_image(img, model_func):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    boxes, probs, labels, *masks = model_func(resized_img)
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [fill_full_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results


def eval_db(df, detect_func, load_annt_func, basedir, names, load_img_annt_func):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        load_annt_func: takes basedir and names
                        and returns dictionary of classes (loads annotations)
        basedir: base directory of the whole data
        names: names of directories containing the data to be evaluated
        load_img_annt_func: takes image ID and the output of 'load_annt_func'
                            and returns annotations (class, bounding box, etc.)

    Returns:
        dict_IoU: dictionary containing lists of IoUs per target class
        total_false_positive: dictionary containing values of FPs per target class
        df.size(): database size
    """
    df.reset_state()
    # define classes for evaluation
    valid_classes = cfg.DATA.CLASS_NAMES[1:]
    # initialize dictionary of lists
    dict_IoU = {}
    img_results = {}
    total_false_positive = {}
    for cls in valid_classes:
        dict_IoU[cls] = []
        img_results[cls] = []
        total_false_positive[cls] = 0
    # initialize and load annotations
    dict_classes = load_annt_func(basedir, names)

    with tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()) as pbar:
        for img, img_id in df.get_data():
            start_time = time.time()
            results = detect_func(img)
            pred_time = time.time() - start_time
            print("Image ID - {}: elapsed time for prediction: {}".format(img_id, pred_time))
            for r in results:
                if r.score < cfg.TEST.RESULT_SCORE_THRESH_VIS:  # exclude
                    continue
                box = r.box
                class_id = r.class_id

                res = {
                    'image_id': img_id,
                    'class_id': class_id,
                    'bbox': list(map(lambda x: round(float(x), 2), box)),
                    'score': round(float(r.score), 3),
                    'IoU_flag': False,
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(
                        np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                img_results[cfg.DATA.CLASS_NAMES[class_id]].append(res)
            img_gt = load_img_annt_func(dict_classes, img_id)
            # img_gt contains only 1 dictionary
            assert len(img_gt) == 1, len(img_gt)
            img_gt = img_gt[0]
            for idx in range(len(img_gt['class'])):
                bbox = img_gt['boxes'][idx]  # bbox is an array consisting of [left, top, right, bottom]
                bbox = list(map(lambda x: round(float(x), 2), bbox))
                target_box = box_shapely(bbox[0], bbox[1], bbox[2], bbox[3])

                IoU_max_val = 0
                # compare relevant bounding boxes to the target bounding box
                for res in img_results[cfg.DATA.CLASS_NAMES[img_gt['class'][idx]]]:
                    res_box = res['bbox']
                    predicted_box = box_shapely(res_box[0], res_box[1], res_box[2], res_box[3])
                    intersection_area = target_box.intersection(predicted_box).area
                    union_area = target_box.union(predicted_box).area
                    IoU_val = intersection_area / union_area
                    # if the calculated value doesn't equal zero => IoU_flag == True
                    if IoU_val > 0:
                        res['IoU_flag'] = True
                    # find maximum IoU value
                    if IoU_val > IoU_max_val:
                        IoU_max_val = IoU_val
                # append the best IoU value for the target
                dict_IoU[cfg.DATA.CLASS_NAMES[img_gt['class'][idx]]].append(IoU_max_val)

            # calculate FPs
            for cls in valid_classes:
                for res in img_results[cls]:
                    # if IoU_flag is False => it's a false detection
                    if not res['IoU_flag']:
                        total_false_positive[cls] += 1
                img_results[cls] = []
            pbar.update(1)
    return dict_IoU, total_false_positive, df.size()


def eval_results(output_file, prediction_results, total_fp, data_size_all, db_names):
    sec_per_hour = 3600
    # define classes for evaluation
    valid_classes = cfg.DATA.CLASS_NAMES[1:]

    # initialize:
    total_size = 0
    predictions_per_class = {}
    fp_per_class = {}
    for class_name in valid_classes:
        predictions_per_class[class_name] = []
        fp_per_class[class_name] = []

    # open file for writing
    res_file = open(output_file, 'w')

    if not isinstance(db_names, (list, tuple)):
        db_names = [db_names]

    for db in db_names:
        predictions_db = []
        fp_db = []
        for class_name in valid_classes:
            predictions_db.extend(prediction_results[db][class_name])
            predictions_per_class[class_name].extend(prediction_results[db][class_name])
            fp_db.append(total_fp[db][class_name])
            fp_per_class[class_name].append(total_fp[db][class_name])
            # calculate statistics for each class in current database:
            # calculate reverse-order cumulative histogram with 10 bins
            hist, _, _ = plt.hist(prediction_results[db][class_name], range=(0, 1), density=True, cumulative=-1)
            print("{} - recall values for class {}: {}".format(db, class_name, hist))
            res_file.write("{} - recall values for class {}: {}".format(db, class_name, hist) + os.linesep)
            false_alarm = (float(total_fp[db][class_name]) / float(data_size_all[db])) * cfg.SYSTEM_FREQUENCY * sec_per_hour
            print("{} - false alarms per hour for class {}: {}".format(db, class_name, false_alarm))
            res_file.write("{} - false alarms per hour for class {}: {}".format(db, class_name, false_alarm) + os.linesep)
        plt.close()  # don't show previous histograms - bug
        # calculate statistics for each database:
        plt.figure()
        # calculate reverse-order cumulative histogram with 10 bins
        hist, _, _ = plt.hist(predictions_db, range=(0, 1), density=True, cumulative=-1)
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel("IoU")
        plt.ylabel("Probability")
        plt.title("{} - recall values".format(db))
        print("{} - recall values: {}".format(db, hist))
        res_file.write("{} - recall values: {}".format(db, hist) + os.linesep)
        false_alarm = (float(sum(fp_db)) / float(data_size_all[db])) * cfg.SYSTEM_FREQUENCY * sec_per_hour
        print("{} - false alarms per hour: {}".format(db, false_alarm))
        res_file.write("{} - false alarms per hour: {}".format(db, false_alarm) + os.linesep)
        total_size += data_size_all[db]

    if len(db_names) > 1:
        predictions = []
        fp = []
        for class_name in valid_classes:
            predictions.extend(predictions_per_class[class_name])
            fp.extend(fp_per_class[class_name])
            # calculate statistics for each class:
            plt.figure()
            # calculate reverse-order cumulative histogram with 10 bins
            hist, _, _ = plt.hist(predictions_per_class[class_name], range=(0, 1), density=True, cumulative=-1)
            plt.xticks(np.linspace(0, 1, 11))
            plt.xlabel("IoU")
            plt.ylabel("Probability")
            plt.title("Total recall values for class {}".format(class_name))
            print("Total recall values for class {}: {}".format(class_name, hist))
            res_file.write("Total recall values for class {}: {}".format(class_name, hist) + os.linesep)
            false_alarm = (float(sum(fp_per_class[class_name])) / float(total_size)) * cfg.SYSTEM_FREQUENCY * sec_per_hour
            print("Total false alarms per hour for class {}: {}".format(class_name, false_alarm))
            res_file.write("Total false alarms per hour for class {}: {}".format(class_name, false_alarm) + os.linesep)
        # calculate total statistics:
        plt.figure()
        # calculate reverse-order cumulative histogram with 10 bins
        hist, _, _ = plt.hist(predictions, range=(0, 1), density=True, cumulative=-1)
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlabel("IoU")
        plt.ylabel("Probability")
        plt.title("Total recall values")
        print("Total recall values: {}".format(hist))
        res_file.write("Total recall values: {}".format(hist) + os.linesep)
        false_alarm = (float(sum(fp)) / float(total_size)) * cfg.SYSTEM_FREQUENCY * sec_per_hour
        print("Total false alarms per hour: {}".format(false_alarm))
        res_file.write("Total false alarms per hour: {}".format(false_alarm))

    res_file.close()
    plt.show()


# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_evaluation_scores(json_file):
    ret = {}
    assert cfg.DATA.BASEDIR and os.path.isdir(cfg.DATA.BASEDIR)
    annofile = os.path.join(
        cfg.DATA.BASEDIR, 'annotations',
        'instances_{}.json'.format(cfg.DATA.VAL))
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.params.catIds = [1, 3]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    for k in range(6):
        ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

    if cfg.MODE_MASK:
        cocoEval = COCOeval(coco, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        for k in range(6):
            ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
    return ret
