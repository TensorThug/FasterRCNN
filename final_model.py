# -*- coding: utf-8 -*-
# File: final_model.py

import cv2
import time
import numpy as np
import six

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

assert six.PY3, "FasterRCNN requires Python 3!"

from tensorpack import *
from tensorpack.tfutils.common import get_tf_version_tuple
import tensorpack.utils.viz as tpviz

from viz import draw_final_outputs
from eval import (
    eval_db, detect_one_image, eval_results, DetectionResult)
from config import finalize_configs, config as cfg
from train import (ResNetC4Model, ResNetFPNModel, DetectionModel)


class FinalModel:
    def __init__(self, full_path_to_model):
        assert full_path_to_model.endswith('.index'), full_path_to_model
        print(get_tf_version_tuple())
        assert get_tf_version_tuple() > (1, 6), "TF<1.6 has a bug which may lead to crash in FasterRCNN training."
        # https://github.com/tensorflow/tensorflow/issues/14657

        self.MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()
        finalize_configs(is_training=False)
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS
        self.pred = OfflinePredictor(PredictConfig(
            model=self.MODEL,
            session_init=get_model_loader(full_path_to_model),
            input_names=self.MODEL.get_inference_tensor_names()[0],
            output_names=self.MODEL.get_inference_tensor_names()[1]))

    def image_pred(self, full_path_to_img):
        start_time = time.time()
        img = cv2.imread(full_path_to_img, cv2.IMREAD_COLOR)
        results = detect_one_image(img, self.pred)
        final = draw_final_outputs(img, results)
        viz = np.concatenate((img, final), axis=1)
        elapsed_time = time.time() - start_time
        print("Elapsed time for prediction: {}".format(elapsed_time))
        tpviz.interactive_imshow(viz)

    def multiple_image_pred(self, paths_to_imgs):
        """
        :param paths_to_imgs: list of full paths to images
        :return: figures with predictions
        """
        for path in paths_to_imgs:
            self.image_pred(path)

