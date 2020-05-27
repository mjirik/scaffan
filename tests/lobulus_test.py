#! /usr/bin/python
# -*- coding: utf-8 -*-

# import logging
# logger = logging.getLogger(__name__)
from loguru import logger
import unittest
import io3d

# import openslide
import scaffan
import scaffan.algorithm
import numpy as np
import os.path as op
import os
import shutil
from pathlib import Path
import scaffan.image
import scaffan.lobulus
import exsu
from unittest.mock import patch
path_to_dir = Path(__file__).parent
import pytest


# def test_run_lobuluses():


class LobulusTest(unittest.TestCase):
    @pytest.mark.slow
    def test_run_lobuluses(self):
        output_dir = (path_to_dir / "test_output/test_lobulus_output_dir").absolute()
        if output_dir.exists():
            shutil.rmtree(output_dir)
            # os.remove(output_dir)
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0165_HE.ndpi", get_root=True)
        # imsl = openslide.OpenSlide(fn)
        # annotations = scan.read_annotations(fn)
        # scan.annotations_to_px(imsl, annotations)
        # Yellow

        original_foo = scaffan.image.AnnotatedImage.get_annotations_by_color
        with patch.object(scaffan.image.AnnotatedImage, 'select_annotations_by_color', autospec=True) as mock_foo:
            def side_effect(sf, annid, *args, **kwargs):
                logger.debug("mocked function select_annotations_by_color()")
                original_list = original_foo(sf, annid, *args, **kwargs)
                logger.debug(f"id={annid}, original ann_ids={original_list}")
                if annid == "#000000":
                    new_list = original_list
                else:
                    new_list = [original_list[0]]
                # print(f"original ann_ids={original_list}")
                logger.debug(f"new ann_ids={new_list}")
                # print(f"new ann_ids={new_list}")
                return new_list

            mock_foo.side_effect = side_effect

            mainapp = scaffan.algorithm.Scaffan()
            mainapp.set_input_file(fn)
            mainapp.set_output_dir(str(output_dir))
            mainapp.init_run()
            mainapp.set_report_level(10)
            mainapp.set_parameter("Input;Lobulus Selection Method", "Color")
            mainapp.set_annotation_color_selection("#FFFF00")
            # mainapp.parameters.param("Processing", "Show").setValue(True)
            mainapp.run_lobuluses()
        logger.debug("imgs: ", mainapp.report.imgs)

        img = mainapp.report.load_array("lobulus_central_thr_skeleton_7.png")
        imsz = np.prod(img.shape)
        lobulus_size = np.sum(img == 1) / imsz
        central_vein_size = np.sum(img == 2) / imsz
        thr_size = np.sum(img == 3) / imsz
        skeleton_size = np.sum(img == 4) / imsz
        self.assertGreater(lobulus_size, 0.10, "Lobulus size 10%")
        self.assertGreater(central_vein_size, 0.001, "Central vein size 0.1%")
        self.assertGreater(thr_size, 0.001, "Threshold size 0.1%")
        self.assertGreater(skeleton_size, 0.001, "Skeleton size 0.1%")
        self.assertGreater(thr_size, skeleton_size, "More threshold than Skeleton")

        self.assertTrue((output_dir / "lobulus_central_thr_skeleton_7.png").exists())
        self.assertTrue(
            (output_dir / "lobulus_central_thr_skeleton_7_skimage.png").exists()
        )
        self.assertTrue((output_dir / "data.xlsx").exists())

        self.assertLess(
            0.6,
            mainapp.evaluation.evaluation_history[0]["Lobulus Border Dice"],
            "Lobulus segmentation should have Dice coefficient above some low level",
        )
        # self.assertLess(0.6, mainapp.evaluation.evaluation_history[1]["Lobulus Border Dice"],
        #                 "Lobulus segmentation should have Dice coefficient above some low level")
        self.assertLess(
            0.2,
            mainapp.evaluation.evaluation_history[0]["Central Vein Dice"],
            "Central Vein segmentation should have Dice coefficient above some low level",
        )
        # mainapp.start_gui()


@pytest.mark.slow
def test_get_lobulus_mask():
    # this is hack to fix the problem with non existing report - not useful anymore
    #
    # output_dir = Path("test_output/test_lobulus_mask_output_dir").absolute()
    # if output_dir.exists():
    #     shutil.rmtree(output_dir)
    # report = exsu.Report(outputdir=output_dir, show=False)

    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    anim = scaffan.image.AnnotatedImage(fn)
    anns = anim.get_annotations_by_color("#0000FF")

    report = None
    lob_proc = scaffan.lobulus.Lobulus(report=report)
    lob_proc.set_annotated_image_and_id(anim, anns[0])
    lob_proc.run()
    # there are several useful masks
    #
    # lob_proc.annotation_mask
    # lob_proc.lobulus_mask
    # lob_proc.central_vein_mask
    # lob_proc.border_mask

    # import matplotlib.pyplot as plt
    # plt.imshow(lob_proc.lobulus_mask)
    # plt.show()

    # this is for testing
    assert np.sum(lob_proc.annotation_mask) > 100, "segmentation should have more than 100 px"
    assert np.sum(lob_proc.lobulus_mask) > 100, "segmentation should have more than 100 px"
    assert np.sum(lob_proc.central_vein_mask) > 0, "segmentation should have more than 0 px"
    assert np.sum(lob_proc.annotation_mask) < np.sum(lob_proc.lobulus_mask), "annotation should be smaller than lobulus"
