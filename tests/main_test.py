#! /usr/bin/python
# -*- coding: utf-8 -*-

# import logging
# logger = logging.getLogger(__name__)
from loguru import logger
import unittest
import os
import io3d


# import openslide
import scaffan
import scaffan.algorithm
from PyQt5 import QtWidgets
import sys
from pathlib import Path
import pytest
from datetime import datetime

qapp = QtWidgets.QApplication(sys.argv)


class MainGuiTest(unittest.TestCase):

    # skip_on_local = True
    skip_on_local = False

    @unittest.skipIf(os.environ.get("TRAVIS", skip_on_local), "Skip on Travis-CI")
    def test_just_start_gui_interactive_with_predefined_params(self):
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0165_HE.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0168_HE.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-003_J-18-0165_HE.ndpi", get_root=True) # training
        fn = io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-002_J-18-0091_HE.ndpi", get_root=True) # training
        # fn = io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-003_J-18-0168_HE.ndpi", get_root=True) # training
        # fn = io3d.datasets.join_path(
        #     "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        # )
        # imsl = openslide.OpenSlide(fn)
        # annotations = scan.read_annotations(fn)
        # scan.annotations_to_px(imsl, annotations)
        mainapp = scaffan.algorithm.Scaffan()
        mainapp.set_input_file(fn)
        # mainapp.set_annotation_color_selection("#FF00FF")
        # mainapp.set_annotation_color_selection("#FF0000")
        mainapp.set_annotation_color_selection("#FFFF00")
        mainapp.set_parameter("Processing;Run Skeleton Analysis", False)
        mainapp.set_parameter("Processing;Run Texture Analysis", False)
        mainapp.set_parameter("Processing;Slide Segmentation;Run Training", True)
        mainapp.set_parameter("Processing;Slide Segmentation;Lobulus Number", 3)
        mainapp.start_gui(qapp=qapp)

    def test_just_start_app(self):
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0165_HE.ndpi", get_root=True)
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        # imsl = openslide.OpenSlide(fn)
        # annotations = scan.read_annotations(fn)
        # scan.annotations_to_px(imsl, annotations)
        mainapp = scaffan.algorithm.Scaffan()
        mainapp.set_input_file(fn)
        # mainapp.set_annotation_color_selection("#FF00FF")
        # mainapp.set_annotation_color_selection("#FF0000")
        # mainapp.set_annotation_color_selection("red")
        mainapp.set_annotation_color_selection("yellow")
        mainapp.start_gui(skip_exec=True, qapp=qapp)

    skip_on_local = True
    @unittest.skipIf(os.environ.get("TRAVIS", True), "Skip on Travis-CI")
    def test_run_lobuluses(self):
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        # imsl = openslide.OpenSlide(fn)
        # annotations = scan.read_annotations(fn)
        # scan.annotations_to_px(imsl, annotations)
        mainapp = scaffan.algorithm.Scaffan()
        mainapp.set_input_file(fn)
        mainapp.set_output_dir("test_run_lobuluses_output_dir")
        # mainapp.init_run()
        # mainapp.set_annotation_color_selection("#FF00FF") # magenta -> cyan
        # mainapp.set_annotation_color_selection("#00FFFF")
        # cyan causes memory fail
        mainapp.set_annotation_color_selection("#FFFF00")
        mainapp.run_lobuluses()
        self.assertLess(0.6, mainapp.evaluation.evaluation_history[0]["Lobulus Border Dice"],
                        "Lobulus segmentation should have Dice coefficient above some low level")
        # self.assertLess(0.6, mainapp.evaluation.evaluation_history[1]["Lobulus Border Dice"],
        #                 "Lobulus segmentation should have Dice coefficient above some low level")
        self.assertLess(0.2, mainapp.evaluation.evaluation_history[0]["Central Vein Dice"],
                        "Central Vein segmentation should have Dice coefficient above some low level")
        # self.assertLess(0.5, mainapp.evaluation.evaluation_history[1]["Central Vein Dice"],
        #                 "Central Vein should have Dice coefficient above some low level")

    skip_on_local = False
    @unittest.skipIf(os.environ.get("TRAVIS", skip_on_local), "Skip on Travis-CI")
    def test_run_lobuluses_manual_segmentation(self):
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        # imsl = openslide.OpenSlide(fn)
        # annotations = scan.read_annotations(fn)
        # scan.annotations_to_px(imsl, annotations)
        mainapp = scaffan.algorithm.Scaffan()
        mainapp.set_input_file(fn)
        mainapp.set_output_dir("test_run_lobuluses_output_dir")
        # mainapp.init_run()
        mainapp.set_annotation_color_selection("#00FFFF")
        mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", True)
        mainapp.run_lobuluses()
        self.assertLess(0.9, mainapp.evaluation.evaluation_history[0]["Lobulus Border Dice"],
                        "Lobulus segmentation should have Dice coefficient above some low level")
        self.assertLess(0.9, mainapp.evaluation.evaluation_history[1]["Lobulus Border Dice"],
                        "Lobulus segmentation should have Dice coefficient above some low level")
        self.assertLess(0.9, mainapp.evaluation.evaluation_history[0]["Central Vein Dice"],
                        "Central Vein segmentation should have Dice coefficient above some low level")
        self.assertLess(0.9, mainapp.evaluation.evaluation_history[1]["Central Vein Dice"],
                        "Central Vein should have Dice coefficient above some low level")

    def test_start_gui_no_exec(self):
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0165_HE.ndpi", get_root=True)
        # imsl = openslide.OpenSlide(fn)
        # annotations = scan.read_annotations(fn)
        # scan.annotations_to_px(imsl, annotations)
        mainapp = scaffan.algorithm.Scaffan()
        mainapp.set_input_file(fn)
        mainapp.set_output_dir("test_output_dir")
        # mainapp.init_run()
        skip_exec = True
        # skip_exec = False
        mainapp.start_gui(skip_exec=skip_exec, qapp=None)


    @pytest.mark.dataset
    @pytest.mark.slow
    def test_training_slide_segmentation_clf(self):
        mainapp = scaffan.algorithm.Scaffan()
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        modtime0 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        logger.debug(f"classificator prior modification time: {modtime0}")
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0165_HE.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0168_HE.ndpi", get_root=True)

        fns = [
            io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-002_J-18-0091_HE.ndpi", get_root=True), # training
            io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-003_J-18-0165_HE.ndpi", get_root=True), # training
            io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-003_J-18-0168_HE.ndpi", get_root=True)  # training
        ]
        for i, fn in enumerate(fns):
            mainapp.set_input_file(fn)
            mainapp.set_output_dir()
            # There does not have to be set some color
            # mainapp.set_annotation_color_selection("#FF00FF")
            # mainapp.set_annotation_color_selection("#FF0000")
            mainapp.set_annotation_color_selection("#FFFF00")
            mainapp.set_parameter("Processing;Run Skeleton Analysis", False)
            mainapp.set_parameter("Processing;Run Texture Analysis", False)
            if i == 0:
                mainapp.set_parameter("Processing;Slide Segmentation;Clean Before Training", True)
            else:
                mainapp.set_parameter("Processing;Slide Segmentation;Clean Before Training", False)
            mainapp.set_parameter("Processing;Slide Segmentation;Run Training", True)
            mainapp.set_parameter("Processing;Slide Segmentation;Lobulus Number", 0)
            # mainapp.start_gui(qapp=qapp)
            mainapp.run_lobuluses()

        assert Path(mainapp.slide_segmentation.clf_fn).exists()
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        modtime1 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        logger.debug(f"classificator prior modification time: {modtime1}")
        assert modtime0 != modtime1



    @pytest.mark.dataset
    @pytest.mark.slow
    def test_testing_slide_segmentation_clf(self):
        mainapp = scaffan.algorithm.Scaffan()
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        modtime0 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        logger.debug(f"classificator prior modification time: {modtime0}")
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0165_HE.ndpi", get_root=True)
        # fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-003_J-18-0168_HE.ndpi", get_root=True)

        fns = [
            io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-003_J-18-0166_HE.ndpi", get_root=True),
            io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-003_J-18-0167_HE.ndpi", get_root=True),
            io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-003_J-18-0169_HE.ndpi", get_root=True)
            # io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-002_J-18-0091_HE.ndpi", get_root=True),
        ]
        for fn in fns:
            mainapp.set_input_file(fn)
            mainapp.set_output_dir()
            # There does not have to be set some color
            # mainapp.set_annotation_color_selection("#FF00FF")
            # mainapp.set_annotation_color_selection("#FF0000")
            mainapp.set_annotation_color_selection("#FFFF00")
            mainapp.set_parameter("Processing;Run Skeleton Analysis", False)
            mainapp.set_parameter("Processing;Run Texture Analysis", False)
            mainapp.set_parameter("Processing;Slide Segmentation;Clean Before Training", False)
            mainapp.set_parameter("Processing;Slide Segmentation;Run Training", False)
            mainapp.set_parameter("Processing;Slide Segmentation;Lobulus Number", 0)
            # mainapp.start_gui(qapp=qapp)
            mainapp.run_lobuluses()

            specimen_size_mm = mainapp.slide_segmentation.sinusoidal_area_mm + mainapp.slide_segmentation.septum_area_mm
            whole_area_mm = mainapp.slide_segmentation.empty_area_mm + specimen_size_mm
            assert specimen_size_mm  > whole_area_mm * 0.1
            assert mainapp.slide_segmentation.sinusoidal_area_mm > 0.1 * specimen_size_mm
            assert mainapp.slide_segmentation.septum_area_mm > 0.1 * specimen_size_mm

        assert Path(mainapp.slide_segmentation.clf_fn).exists()
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        modtime1 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        logger.debug(f"classificator prior modification time: {modtime1}")
        assert modtime0 == modtime1
