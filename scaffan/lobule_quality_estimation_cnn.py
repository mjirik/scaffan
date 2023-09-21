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

# from pprint import pprint, pformat
# import time

# import cv2 # pokud potřebujeme jen měnit velikost, raději bych cv2 ze závislostí vynechal
import skimage.transform

# from statistics import mean

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
        pname="SNI Prediction CNN",
        # ptype="group",
        ptype="bool",
        pvalue=True,
        # pvalue=False,
        ptip="CNN estimator of quality",
    ):
        params = [
            {
                "name": "Version",
                "type": "str",
                "value": "v1",
                # "suffix": "px",
                "siPrefix": False,
                "tip": "Version of used CNN model",
            },
            # {
            #     "name": "Example Float Param",
            #     "type": "float",
            #     "value": 0.00006,
            #     "suffix": "m",
            #     "siPrefix": True,
            #     "tip": "Value defines size of something",
            # },
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

    def init(self, force_download_model=False):
        import tensorflow

        logger.debug(
            f"tensorflow version={tensorflow.__version__}, pth={tensorflow.__file__}"
        )
        from tensorflow.keras.models import load_model

        # načtení architektury modelu
        # načtení parametrů modelu

        cnn_model_version = str(self.parameters.param("Version").value())
        model_path = self._get_devel_model_path()
        if not model_path.exists() or force_download_model:
            model_path = self.download_model(cnn_model_version)
        logger.debug(
            f"model_path[{str(type(model_path))}]={model_path} ; exists={model_path.exists()}"
        )
        # model_path = str(model_path)
        # logger.debug(f"model_path[{type(model_path)}:{model_path}")
        # TODO fix the problem with cuda
        self.model = load_model(str(model_path))

    def _get_devel_model_path(self):
        cnn_model_version = str(self.parameters.param("Version").value())
        model_path = path_to_script / f"{cnn_model_version}.h5"
        return model_path

    def download_model(self, cnn_model_version):
        import requests

        model_path = Path(f"~/.scaffan/{cnn_model_version}.h5").expanduser()
        url = f"https://github.com/mjirik/scaffan/raw/master/scaffan/{cnn_model_version}.h5"
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Downloading from '{url}' to {str(model_path)}")
            r = requests.get(url, allow_redirects=True)
            open(model_path, "wb").write(r.content)
        return model_path

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
        # snip_corners = self.cut_the_image(
        #     self.lobulus_segmentation, self.parent_view.region_pixelsize[0], False
        # )
        snip_corners = self.cut_the_image2(
            self.lobulus_segmentation, self.parent_view.region_pixelsize[0]
        )

        crop_size = int((1 / self.parent_view.region_pixelsize[0]) * self.CUT_SIZE)
        evaluations = []

        # Phase 2: Predict SNI value for every sample
        for i, cut_point in enumerate(snip_corners):
            try:
                img = self.parent_view.get_region_image(as_gray=True)
                image_snip = img[
                    cut_point[0] : cut_point[0] + crop_size,
                    cut_point[1] : cut_point[1] + crop_size,
                ]
                image_snip = self.shrink_image(image_snip)
                prediction = self.model.predict(
                    image_snip.reshape(1, self.DISPLAY_SIZE, self.DISPLAY_SIZE, 1),
                    verbose=0,
                )
                # Predictions are normalized to an interval <0,1> but SNI belongs to <0,2>
                sni_prediction = 2 * np.float(prediction)
                evaluations.append(sni_prediction)
            except ValueError:
                import traceback
                logger.error(traceback.format_exc())
                logger.debug(f"crop_size={crop_size}")
                logger.debug(f"img.shape={img.shape}")
                logger.debug(f"cut_point={cut_point}")

        # výsledek uložený do proměnné sni_prediction

        sni_prediction = np.mean(evaluations) if len(evaluations) > 0 else np.NaN

        if self.report:
            label = "SNI prediction CNN"
            self.report.actual_row[label] = sni_prediction
        return sni_prediction

    def cut_the_image2(self, mask, pixel_size):
        """
        Morphological solver for splitting the lobule area.
        M. Jirik
        """
        cut_pixel_size = int((1 / pixel_size) * self.CUT_SIZE)
        step_size = int(cut_pixel_size / self.STEPS_PER_CUT)
        half_size = int(cut_pixel_size / 2)

        corner_list = []
        msk = skimage.morphology.binary_erosion(
            mask, skimage.morphology.square(cut_pixel_size)
        )
        # remove borders
        msk[:half_size, :] = False
        msk[-half_size:, :] = False
        msk[:, half_size] = False
        msk[:, -half_size:] = False

        aa0, aa1 = np.nonzero(msk)
        is_there_pixel = len(aa0) > 0
        while is_there_pixel:
            p0 = aa0[0]
            p1 = aa1[0]

            min0 = max(p0 - cut_pixel_size, 0)
            max0 = min(p0 + cut_pixel_size, msk.shape[0])
            min1 = max(p1 - cut_pixel_size, 0)
            max1 = min(p1 + cut_pixel_size, msk.shape[1])
            corner_list.append([p0 - half_size, p1 - half_size])

            msk[min0:max0, min1:max1] = False
            aa0, aa1 = np.nonzero(msk)
            is_there_pixel = len(aa0) > 0
        return corner_list

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
                if x < mask.shape[1] and y < mask.shape[0]:
                    mask[y, x] = 5
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
        return skimage.transform.resize(
            img,
            output_shape=(self.DISPLAY_SIZE, self.DISPLAY_SIZE),
            preserve_range=True,
        )
        # pokud nepotřebujeme z cv2 nic jiného, zkusil bych jej nahradit, aby se nezvětšovaly závislosti.
        # return cv2.resize(img, dsize=(self.DISPLAY_SIZE, self.DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
