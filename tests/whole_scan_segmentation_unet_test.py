#! /usr/bin/python
# -*- coding: utf-8 -*-

# import logging
# logger = logging.getLogger(__name__)
from loguru import logger
import unittest
import os
import os.path as op
import sys
import io3d
from pathlib import Path

path_to_script = op.dirname(op.abspath(__file__))
sys.path.insert(0, op.abspath(op.join(path_to_script, "../../exsu")))
# sys.path.insert(0, op.abspath(op.join(path_to_script, "../../imma")))
exsu_pth = Path(__file__).parents[2] / "exsu"
logger.debug(f"exsupth{exsu_pth}, {exsu_pth.exists()}")
sys.path.insert(0, str(exsu_pth))

import exsu

logger.debug(f"exsu path: {exsu.__file__}")
import numpy as np

# import openslide
import scaffan
import scaffan.algorithm

# from PyQt5 import QtWidgets
import pytest

# from datetime import datetime
import scaffan.image
import scaffan.whole_slide_seg_unet

# qapp = QtWidgets.QApplication(sys.argv)

# tohle říká, že test může/musí selhat
# @pytest.mark.xfail
def test_unet_on_view():
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    anim = scaffan.image.AnnotatedImage(fn)
    ann_ids = anim.get_annotations_by_color("#FFFF00")
    # anim.get_views(ann_ids)
    view = anim.get_view(
        annotation_id=ann_ids[0], size_on_level=[224, 224], pixelsize_mm=[0.01, 0.01]
    )
    # import matplotlib.pyplot as plt
    # im = view.get_region_image()
    # plt.imshow(im)
    # plt.show()
    # assert np.array_equal(im.shape, [224, 224, 4])
    wss_unet = scaffan.whole_slide_seg_unet.WholeSlideSegmentationUNet()
    wss_unet.init_segmentation()
    prediction = wss_unet.predict_tile(view)
    # plt.imshow(prediction)
    # plt.show()

    unq = np.unique(prediction)
    assert 0 in unq, "label 0 should be in prediction"
    assert 1 in unq, "label 1 should be in prediction"
    # assert 2 in unq, "label 1 should be in prediction"


def test_unet_on_view_czi():
    # fn = io3d.datasets.join_path(
    #     "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    # )
    fn = io3d.datasets.join_path(
        "medical/orig/scaffan-analysis-czi/Zeiss-scans/01_2019_11_12__RecognizedCode.czi",
        get_root=True,
    )
    anim = scaffan.image.AnnotatedImage(fn)
    # ann_ids = anim.get_annotations_by_color("#FFFF00")
    # anim.get_views(ann_ids)
    view = anim.get_view(
        location_mm=[4.0, 4.0], size_on_level=[224, 224], pixelsize_mm=[0.01, 0.01]
    )
    import matplotlib.pyplot as plt

    im = view.get_region_image()
    # plt.imshow(im)
    # plt.show()
    # assert np.array_equal(im.shape, [224, 224, 4])
    wss_unet = scaffan.whole_slide_seg_unet.WholeSlideSegmentationUNet()
    wss_unet.init_segmentation()
    prediction = wss_unet.predict_tile(view)
    im = view.get_region_image()
    height0 = anim.openslide.properties["openslide.level[0].height"]
    width0 = anim.openslide.properties["openslide.level[0].width"]
    loc = view.region_location
    logger.debug(f"loc={loc}, size={(height0, width0)}")
    im = view.get_region_image()
    plt.figure()
    plt.imshow(im)
    plt.contour(prediction)
    # plt.show()

    unq = np.unique(prediction)
    assert 0 in unq, "label 0 should be in prediction"
    assert 1 in unq, "label 1 should be in prediction"
    # assert False
    # assert 2 in unq, "label 1 should be in prediction"
