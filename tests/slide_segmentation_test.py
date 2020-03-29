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

import scaffan.slide_segmentation


def test_slide_segmentation():
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    seg = scaffan.slide_segmentation.ScanSegmentation()
    dir(seg)
    anim = scaffan.image.AnnotatedImage(fn)

    seg.init(anim)
    seg.parameters.param("Segmentation Method").setValue("HCFTS")
    ann_ids_black = seg.anim.select_annotations_by_color("#000000")
    assert 10 in ann_ids_black
    assert 11 in ann_ids_black
    seg.run()
    assert seg.full_raster_image.shape[:2] == seg.full_output_image.shape[:2]
    plt.imshow(seg.full_output_image)
    plt.show()
    plt.imshow(seg.full_raster_image)
    plt.show()
    # seg._find_best_level()
    # seg._find_best_level()
