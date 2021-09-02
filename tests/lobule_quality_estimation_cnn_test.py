#! /usr/bin/python
# -*- coding: utf-8 -*-

from loguru import logger
import unittest
import io3d

# import openslide
import scaffan
import scaffan.lobulus
# import scaffan.algorithm
import scaffan.lobule_quality_estimation_cnn
import pytest
import re
import numpy as np
import pdb


def test_get_lobulus_mask_manual():
    fn = io3d.datasets.join_path(
        # "medical/orig/scaffan-analysis-czi/Zeiss-scans/01_2019_11_12__RecognizedCode_crop.czi",
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi",
        get_root=True
    )
    # logger.debug(f"fn.exists={fn.exists()}")

    # Get the manual segmentation

    anim = scaffan.image.AnnotatedImage(fn)
    anns = anim.get_annotations_by_color("#FFFF00")
    # anns = anim.get_annotations_by_color("#FF0000")

    report = None
    lob_proc = scaffan.lobulus.Lobulus(report=report)
    lob_proc.parameters.param("Manual Segmentation").setValue(True)
    annotation_id = anns[0]
    lob_proc.set_annotated_image_and_id(anim, annotation_id=annotation_id)
    lob_proc.run()

    # make the calculation faster by using about quater of the lobule
    sh_half = (np.asarray(lob_proc.lobulus_mask.shape) / 2).astype(int)
    lob_proc.lobulus_mask[:sh_half[0], :sh_half[1]] = False

    lqe = scaffan.lobule_quality_estimation_cnn.LobuleQualityEstimationCNN()


    lqe.init(force_download_model=True) # this will test the model download
    lqe.set_input_data(
        view=lob_proc.view,
        annotation_id=annotation_id,
        lobulus_segmentation=lob_proc.lobulus_mask,
    )
    quality = lqe.run()

    reg_search = re.search("SNI=(\d*\.?\d*) ", lob_proc.anim.annotations[annotation_id]['details'])

    expected_quality = float(reg_search.group(1))
    expected_quality_range = 0.1
    assert quality == pytest.approx(expected_quality, expected_quality_range)
