# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import pytest
import scaffan
import io3d  # just to get data
import scaffan.image as scim
from typing import List
import exsu
from matplotlib import pyplot as plt
import numpy as np
# file_path = Path()

import scaffan.slide_segmentation


def test_slide_segmentation():
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    report = exsu.Report(outputdir="./tests/slide_seg_test_output/")
    seg = scaffan.slide_segmentation.ScanSegmentation(report=report)
    # dir(seg)
    anim = scaffan.image.AnnotatedImage(fn)

    seg.init(anim)
    seg.parameters.param("Segmentation Method").setValue("HCFTS")
    seg.parameters.param("Save Training Labels").setValue(True)
    ann_ids_black = seg.anim.select_annotations_by_color("#000000")
    assert 10 in ann_ids_black
    assert 11 in ann_ids_black
    seg.run()
    assert type(seg.full_output_image) == np.ndarray
    assert type(seg.full_raster_image) == np.ndarray
    assert type(seg.whole_slide_training_labels) == np.ndarray
    assert seg.full_raster_image.shape[:2] == seg.full_output_image.shape[:2]
    assert seg.whole_slide_training_labels.shape[:2] == seg.full_output_image.shape[:2]
    # plt.imshow(seg.full_output_image)
    # plt.show()
    # plt.imshow(seg.full_raster_image)
    # plt.show()
    # plt.imshow(seg.whole_slide_training_labels)
    # plt.show()
