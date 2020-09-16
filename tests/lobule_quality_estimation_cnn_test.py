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

def test_get_lobulus_mask_manual():
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )

    # Get the manual segmentation
    anim = scaffan.image.AnnotatedImage(fn)
    anns = anim.get_annotations_by_color("#FFFF00")

    report = None
    lob_proc = scaffan.lobulus.Lobulus(report=report)
    lob_proc.parameters.param("Manual Segmentation").setValue(True)
    annotation_id = anns[0]
    lob_proc.set_annotated_image_and_id(anim, annotation_id=annotation_id)
    lob_proc.run()

    # there are several useful masks
    #
    # lob_proc.annotation_mask
    # lob_proc.lobulus_mask
    # lob_proc.central_vein_mask
    # lob_proc.border_mask

    lqe = scaffan.lobule_quality_estimation_cnn.LobuleQualityEstimationCNN()

    # TODO uncomment this

    # lqe.init()
    # lqe.set_input_data(
    #     view=lob_proc.view,
    #     annotation_id=annotation_id,
    #     lobulus_segmentation=lob_proc.lobulus_mask,
    # )
    # quality = lqe.run()
    # #
    # # # TODO now the acceptance range is too wide. Set the values closer.
    # expected_quality = 0.7
    # expected_quality_range = 0.3
    # assert quality == pytest.approx(expected_quality, expected_quality_range)

