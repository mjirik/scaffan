from pyqtgraph.parametertree import Parameter
from loguru import logger
from . import image
from exsu import Report
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import os
from pathlib import Path
import time

# import cv2 # pokud potřebujeme jen měnit velikost, raději bych cv2 ze závislostí vynechal
import skimage.transform
from statistics import mean

#
# The automatic test is in
# main_test.py: test_testing_slide_segmentation_clf_unet()
path_to_script = Path(os.path.dirname(os.path.abspath(__file__)))
# path_to_scaffan = Path(os.path.join(path_to_script, ".."))




class LobuleQualityEstimationCNN:
    CUT_SIZE = 0.2  # Size of training data [mm]
    STEPS_PER_CUT = 4
    DISPLAY_SIZE = 80  # [px]

    def __init__(
        self,
        report: Report = None,
        pname="Quality Estimation CNN",
        # ptype="group",
        ptype="bool",
        pvalue=False,
        ptip="CNN estimator of quality",
    ):
        params = [
            {
                "name": "Example Integer Param",
                "type": "int",
                "value": 224,
                "suffix": "px",
                "siPrefix": False,
                "tip": "Value defines size of something",
            },
            {
                "name": "Example Float Param",
                "type": "float",
                "value": 0.00006,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Value defines size of something",
            },
        ]
        self.parameters = Parameter.create(
            name=pname,
            type=ptype,
            value=pvalue,
            tip=ptip,
            children=params,
            expanded=False,
        )
        if report is None:
            report = Report()
            report.save = False
            report.show = False
        self.report: Report = report
        self.model = None
        pass

    def init(self):
        from tensorflow.keras.models import load_model
        # načtení architektury modelu
        # načtení parametrů modelu

        model_path = path_to_script / "cnn models" / "v7.h5"
        logger.debug(f"model_path[{type(model_path)}={model_path}")
        # model_path = str(model_path)
        # logger.debug(f"model_path[{type(model_path)}:{model_path}")
        # TODO fix the problem with cuda
        self.model = load_model(str(model_path))

    def set_input_data(
            self, view: image.View, annotation_id: int = None, lobulus_segmentation=None
    ):
        self.anim = view.anim
        self.annotation_id = annotation_id
        self.parent_view = view
        logger.trace(f"lobulus segmentation {lobulus_segmentation}")
        self.lobulus_segmentation = lobulus_segmentation

    def run(self):
        # Tady by bylo tělo algoritmu

        # Phase 1: Cut out rectangle samples from original image
        snip_corners = self.cut_the_image(self.lobulus_segmentation, self.parent_view.region_pixelsize[0], False)
        crop_size = int((1 / self.parent_view.region_pixelsize[0]) * self.CUT_SIZE)
        evaluations = []

        # Phase 2: Predict SNI value for every sample
        for i, cut_point in enumerate(snip_corners):
            img = self.parent_view.get_region_image(True)
            image_snip = img[cut_point[0]:cut_point[0] + crop_size, cut_point[1]:cut_point[1] + crop_size]
            image_snip = self.shrink_image(image_snip)
            prediction = self.model.predict(image_snip.reshape(1, self.DISPLAY_SIZE, self.DISPLAY_SIZE, 1), verbose=0)
            # Predictions are normalized to an interval <0,1> but SNI belongs to <0,2>
            sni_prediction = 2 * np.float(prediction)
            evaluations.append(sni_prediction)

        # výsledek uložený do proměnné sni_prediction
        sni_prediction = mean(evaluations)

        if self.report:
            label = "SNI prediction CNN"
            self.report.actual_row[label] = sni_prediction
        return sni_prediction

    def cut_the_image(self, mask, pixel_size, overlap=True):
        """
        Returns list of coordinates of left upper corners of square samples which fit in the mask.
        """
        cut_pixel_size = int((1 / pixel_size) * self.CUT_SIZE)
        step_size = int(cut_pixel_size / self.STEPS_PER_CUT)

        corner_list = []

        x = 0

        while x < mask.shape[1]:
            skip_next_line = False

            y = 0

            while y < mask.shape[0]:
                if self.does_the_square_fit(mask, x, y, step_size):
                    corner_list.append([y, x])

                    if overlap:
                        y = y + step_size
                    else:
                        y = y + cut_pixel_size
                        skip_next_line = True

                else:
                    y = y + step_size

            if skip_next_line:
                x = x + cut_pixel_size
            else:
                x = x + step_size

        return corner_list

    def does_the_square_fit(self, mask, x_start, y_start, step_size) -> bool:
        """
        Returns True if the square snip fits in the mask
        """
        if mask.shape[0] <= x_start + self.STEPS_PER_CUT * step_size:
            return False

        if mask.shape[1] <= y_start + self.STEPS_PER_CUT * step_size:
            return False

        for i in range(self.STEPS_PER_CUT + 1):
            x = x_start + i * step_size
            y = y_start
            if not mask[x, y]:
                return False

        for i in range(self.STEPS_PER_CUT + 1):
            x = x_start + self.STEPS_PER_CUT * step_size
            y = y_start + i * step_size
            if not mask[x, y]:
                return False

        for i in range(self.STEPS_PER_CUT + 1):
            x = x_start + i * step_size
            y = y_start + self.STEPS_PER_CUT * step_size
            if not mask[x, y]:
                return False

        for i in range(self.STEPS_PER_CUT + 1):
            x = x_start
            y = y_start + i * step_size
            if not mask[x, y]:
                return False

        return True

    def shrink_image(self, img):
        """
        Resize image to display size
        """
        return skimage.transform.resize(img, output_shape=(self.DISPLAY_SIZE, self.DISPLAY_SIZE), preserve_range=True)
        # pokud nepotřebujeme z cv2 nic jiného, zkusil bych jej nahradit, aby se nezvětšovaly závislosti.
        # return cv2.resize(img, dsize=(self.DISPLAY_SIZE, self.DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
