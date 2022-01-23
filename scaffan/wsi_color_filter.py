# /usr/bin/env python
# -*- coding: utf-8 -*-

import re
import re
import sklearn.mixture
import sklearn.neighbors
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
import scaffan.image_intensity_rescale
from . import image_intensity_rescale
from pyqtgraph.parametertree import Parameter
from . import image
from loguru import logger


class WsiColorFilterPQG:
    def __init__(
        self,
        pname="Color Filter",
        ptype="bool",
        # ptype="group",
        pvalue=True,
        ptip="A preprocessing of input image. Turns color specified by annotation into color sepcified in annotation title."
        + "The annotation title must contain 'convert color to #ffffff' to turn color to white.",
        pexpanded=False,
    ):
        # self.rescale_intensity_percentile = image_intensity_rescale.RescaleIntensityPercentile()
        params = [
            # {
            #         "name": "Run Intensity Normalization",
            #         "type": "bool",
            #         "tip": "Do the histogram normalization",
            #         "value": False
            #     },
            {
                "name": "Sigmoidal Slope",
                "type": "float",
                "tip": "Slope of sigmoidal limit function. The lower number means softer application of color change.",
                "value": 0.2,
            },
            # {
            #     "name": "Sigmoidal Slope",
            #     "type": "float",
            #     "tip": "Slope of sigmoidal limit function. The lower number means softer application of color change.",
            #     "value": 0.5,
            # },
        ]
        self.parameters = Parameter.create(
            name=pname,
            type=ptype,
            value=pvalue,
            tip=ptip,
            children=params,
            expanded=pexpanded,
        )
        self.wsi_color_filter = WsiColorFilter()

    def set_anim_params(self, anim: image.AnnotatedImage):
        """
        Set parametetrs of AnnotatedImage to intensity rescale.
        :param anim:
        :return:
        """
        # int_norm_params = self.parameters.param("Processing", "Intensity Normalization")
        # int_norm_params = self.parameters
        # run_resc_int = int_norm_params.param("Run Intensity Normalization").value()
        # run_resc_int = self.parameters.param("Processing", "Intensity Normalization").value()
        # run_resc_int = self.parameters.value()
        self.wsi_color_filter.init_color_filter_by_anim(anim)
        self.wsi_color_filter.slope = (
            self.parameters.param("Sigmoidal Slope").value(),
        )


class WsiColorFilter:
    def __init__(self):
        self.models = {}
        self.color_ids = {}
        self.slope = 1
        self.proportion = 1.
        pass

    def init_color_filter_by_anim(self, anim: image.AnnotatedImage):
        logger.trace(anim.id_by_titles)
        regex = " *convert *color *to *(#[a-fA-F0-9]{6})"
        ids = anim.select_annotations_by_title_regex(regex)
        logger.trace(ids)

        color_hexacodes = [
            re.findall(regex, anim.annotations[id]["title"])[0] for id in ids
        ]

        self.color_ids = {
            value: number for number, value in enumerate(np.unique(color_hexacodes))
        }

        chsv_data = []

        # Get pixels from all annotations in one table. Code the output colors into numbers according to `color_ids`

        # id = ids[0]
        for id in ids:
            views = anim.get_views(annotation_ids=[id])
            mask = views[0].get_annotation_region_raster(id)
            color_hexacode = re.findall(regex, anim.annotations[id]["title"])[0]

            from skimage.color import rgb2hsv

            # view = anim.get_view(
            #     location_mm=[10, 11],
            #     size_on_level=size_on_level,
            #     level=5,
            #     # size_mm=[0.1, 0.1]
            # )
            img = views[0].get_region_image(as_gray=False)
            img_chsv = rgb_image_to_chsv(img)
            datai = img_chsv[mask > 0]
            # codes = np.zeros([1,datai.shape[1]])
            # codes[:] = color_ids[color_hexacode]
            codes = np.array([self.color_ids[color_hexacode]] * datai.shape[0]).reshape(
                [-1, 1]
            )
            datai = np.concatenate((datai, codes), axis=1)

            chsv_data.append(datai)

        if len(chsv_data) > 0:
            chsv_data = np.concatenate(chsv_data, axis=0)

        # Do the training
        # models = {}
        for hexa, color_id in self.color_ids.items():
            chsv_data_i = chsv_data[chsv_data[:, -1] == color_id][:, :-1]
            model = sklearn.mixture.BayesianGaussianMixture()
            # model = sklearn.neighbors.KernelDensity()

            model.fit(chsv_data_i)
            self.models[hexa] = model
        pass

    def img_processing(self, img: np.ndarray, return_proba=False) -> np.ndarray:
        if len(self.models) > 0:
            img_copy = img.copy()

        proba = {}
        for hexa in self.models:
            img_chsv = rgb_image_to_chsv(img_copy)
            sh = img_chsv.shape
            # flatten the image (roll the x and y axes to the end, squeeze, roll back)
            chsv_data2 = np.moveaxis(
                np.moveaxis(img_chsv, 2, 0).reshape([sh[2], -1]), 0, 1
            )
            # chsv_proba = model.predict_proba(chsv_data2)
            chsv_proba2 = self.models[hexa].score_samples(chsv_data2)
            #
            # chsv_proba_img = chsv_proba.reshape(sh[:2])
            chsv_proba2_img = chsv_proba2.reshape(sh[:2])
            # slope = 1.

            #    limit under zero and suqeeze data
            #    log(exp(log(score)) + 1)
            weighted_chsv_proba2_img = self._weighted_proba(chsv_proba2_img)

            #
            if return_proba:
                proba[hexa] = chsv_proba2_img
            img = change_color_using_probability(img, weighted_chsv_proba2_img, hexa)

        if return_proba:
            return img, proba
        else:
            return img

    def _weighted_proba(self, chsv_proba2_img):
        chsv_proba2_img_exp = np.exp(chsv_proba2_img)
        return self._soft_maximum_limit(chsv_proba2_img_exp)

    def _soft_maximum_limit(self, chsv_proba2_img_exp):
        # chsv_proba2_img = scaffan.image_intensity_rescale.sigmoidal(np.log(chsv_proba2_img_exp + 1))
        # chsv_proba2_img = 1 - (np.exp(-np.log(chsv_proba2_img_exp + 1) )) ## this seems to be ok
        chsv_proba2_img_log = np.log(chsv_proba2_img_exp + 1)
        # chsv_proba2_img = 1 / (1 + np.exp(-(chsv_proba2_img_exp)))
        chsv_proba2_img_out = 1 - np.exp(-chsv_proba2_img_log * self.slope)  # best
        # chsv_proba2_img = 1-np.exp(-chsv_proba2_img* slope)
        # chsv_proba2_img = chsv_proba2_img_exp* slope
        # chsv_proba2_img[chsv_proba2_img > 1.] = 1
        x = chsv_proba2_img_exp
        chsv_proba2_img_out = self.proportion * (1 - np.exp(-x / self.slope))
        return chsv_proba2_img_out


def rgb_image_to_chsv(img):
    img_hsv = rgb2hsv(img[:, :, :3])
    img_chsv = hue_to_continuous_2d(img_hsv)
    return img_chsv


def hue_to_continuous_2d(img):
    """Takes hsv image and returns ´hhsv´ format, where hue is replaced by 2 values - sin(hue) and cos(hue)"""
    hue = np.expand_dims(img[:, :, 0], -1)
    hue_x = np.cos(hue * 2 * np.pi)
    hue_y = np.sin(hue * 2 * np.pi)
    img = np.concatenate((hue_x, hue_y, img[:, :, 1:]), axis=-1)
    return img


def change_color_using_probability(img_rgb, img_proba, target_color):
    import matplotlib.colors

    # target_color = '#B4FBB8'
    color_rgb = np.asarray(matplotlib.colors.to_rgb(target_color))
    color_hsv = rgb2hsv(color_rgb.reshape([1, 1, 3]))
    img_hsv = rgb2hsv(img_rgb)
    diff = color_hsv - img_hsv

    # img_hsv - img_hsv - color_hsv
    if img_proba.ndim == 2:
        img_proba = np.expand_dims(img_proba, 2)
    elif img_proba.ndim == 3:
        pass
    else:
        raise ValueError("probability is expected to be 2D or 3D")

    img_proba3 = np.concatenate([img_proba] * 3, axis=2)

    new_img = hsv2rgb(img_hsv + img_proba3 * diff)
    return new_img
