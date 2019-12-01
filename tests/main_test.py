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
sys.path.insert(0, exsu_pth)

import exsu

logger.debug(f"exsu path: {exsu.__file__}")
# import openslide
import scaffan
import scaffan.algorithm
from PyQt5 import QtWidgets
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
        fn = io3d.datasets.join_path(
            "medical",
            "orig",
            "Scaffan-analysis",
            "PIG-002_J-18-0091_HE.ndpi",
            get_root=True,
        )  # training
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
        mainapp.set_parameter("Processing;Skeleton Analysis", False)
        mainapp.set_parameter("Processing;Texture Analysis", False)
        mainapp.set_parameter("Processing;Scan Segmentation;Run Training", True)
        mainapp.set_parameter("Processing;Scan Segmentation;Lobulus Number", 3)
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
        mainapp.set_parameter(
            "Processing;Lobulus Segmentation;Manual Segmentation", True
        )
        mainapp.run_lobuluses()
        self.assertLess(
            0.9,
            mainapp.evaluation.evaluation_history[0]["Lobulus Border Dice"],
            "Lobulus segmentation should have Dice coefficient above some low level",
        )
        self.assertLess(
            0.9,
            mainapp.evaluation.evaluation_history[1]["Lobulus Border Dice"],
            "Lobulus segmentation should have Dice coefficient above some low level",
        )
        self.assertLess(
            0.9,
            mainapp.evaluation.evaluation_history[0]["Central Vein Dice"],
            "Central Vein segmentation should have Dice coefficient above some low level",
        )
        self.assertLess(
            0.9,
            mainapp.evaluation.evaluation_history[1]["Central Vein Dice"],
            "Central Vein should have Dice coefficient above some low level",
        )

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

        fns = [
            io3d.datasets.join_path(
                "medical",
                "orig",
                "Scaffan-analysis",
                "PIG-002_J-18-0091_HE.ndpi",
                get_root=True,
            ),  # training
            io3d.datasets.join_path(
                "medical",
                "orig",
                "Scaffan-analysis",
                "PIG-003_J-18-0165_HE.ndpi",
                get_root=True,
            ),  # training
            io3d.datasets.join_path(
                "medical",
                "orig",
                "Scaffan-analysis",
                "PIG-003_J-18-0168_HE.ndpi",
                get_root=True,
            ),  # training
            # io3d.datasets.join_path("medical", "orig", "Scaffan-analysis", "PIG-003_J-18-0169_HE.ndpi", get_root=True)  # training  bubles
        ]
        self._slide_segmentation_train_clf(fns)

    def test_training_small_slide_segmentation_clf(self):

        fns = [
            io3d.datasets.join_path(
                "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
            ),
        ]
        self._slide_segmentation_train_clf(fns, clf_fn=".temp_clf.pkl")

    @pytest.mark.dataset
    @pytest.mark.slow
    def test_testing_slide_segmentation_clf(self):
        fns = [
            io3d.datasets.join_path(
                "medical",
                "orig",
                "Scaffan-analysis",
                "PIG-003_J-18-0166_HE.ndpi",
                get_root=True,
            ),
            io3d.datasets.join_path(
                "medical",
                "orig",
                "Scaffan-analysis",
                "PIG-003_J-18-0167_HE.ndpi",
                get_root=True,
            ),
            io3d.datasets.join_path(
                "medical",
                "orig",
                "Scaffan-analysis",
                "PIG-003_J-18-0169_HE.ndpi",
                get_root=True,
            )
            # io3d.datasets.join_path("medical", "orig","Scaffan-analysis", "PIG-002_J-18-0091_HE.ndpi", get_root=True),
        ]
        self._testing_slide_segmentation_clf(fns)

    def test_testing_slide_segmentation_clf(self):
        fns = [
            io3d.datasets.join_path(
                "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
            ),
        ]
        self._testing_slide_segmentation_clf(fns)

    def _testing_slide_segmentation_clf(self, fns):

        mainapp = scaffan.algorithm.Scaffan()
        # if clf_fn is not None:
        #     mainapp.slide_segmentation.clf_fn = Path(clf_fn)
        # clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        clf_fn = mainapp.slide_segmentation.clf_fn
        assert clf_fn.exists()

        if clf_fn.exists():
            modtime0 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        else:
            modtime0 = ""
        logger.debug(f"classificator prior modification time: {modtime0}")

        for fn in fns:
            mainapp.set_input_file(fn)
            mainapp.set_output_dir()
            # There does not have to be set some color
            # mainapp.set_annotation_color_selection("#FF00FF")
            # mainapp.set_annotation_color_selection("#FF0000")
            mainapp.set_annotation_color_selection("#FFFF00")
            mainapp.set_parameter("Input;Automatic Lobulus Selection", True)
            mainapp.set_parameter("Processing;Skeleton Analysis", False)
            mainapp.set_parameter("Processing;Texture Analysis", False)
            mainapp.set_parameter("Processing;Open output dir", False)
            mainapp.set_parameter(
                "Processing;Scan Segmentation;Clean Before Training", False
            )
            mainapp.set_parameter("Processing;Scan Segmentation;Run Training", False)
            mainapp.set_parameter("Processing;Scan Segmentation;Lobulus Number", 0)
            # mainapp.start_gui(qapp=qapp)
            mainapp.run_lobuluses()

            specimen_size_mm = (
                mainapp.slide_segmentation.sinusoidal_area_mm
                + mainapp.slide_segmentation.septum_area_mm
            )
            whole_area_mm = mainapp.slide_segmentation.empty_area_mm + specimen_size_mm
            logger.debug("asserts")
            assert specimen_size_mm > whole_area_mm * 0.1
            assert (
                mainapp.slide_segmentation.sinusoidal_area_mm > 0.1 * specimen_size_mm
            )
            assert mainapp.slide_segmentation.septum_area_mm > 0.1 * specimen_size_mm

        assert Path(mainapp.slide_segmentation.clf_fn).exists()
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        modtime1 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        logger.debug(f"classificator prior modification time: {modtime1}")
        assert modtime0 == modtime1

    def _slide_segmentation_train_clf(self, fns, clf_fn=None):
        mainapp = scaffan.algorithm.Scaffan()
        if clf_fn is not None:
            mainapp.slide_segmentation.clf_fn = clf_fn
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        logger.debug(f"clf_fn={clf_fn}")
        if clf_fn.exists():
            modtime0 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        else:
            modtime0 = ""
        logger.debug(f"classificator prior modification time: {modtime0}")
        mainapp.train_scan_segmentation(fns)

        # for i, fn in enumerate(fns):
        #     mainapp.set_input_file(fn)
        #     mainapp.set_output_dir()
        #     # There does not have to be set some color
        #     # mainapp.set_annotation_color_selection("#FF00FF")
        #     # mainapp.set_annotation_color_selection("#FF0000")
        #     mainapp.set_annotation_color_selection("#FFFF00")
        #     mainapp.set_parameter("Input;Automatic Lobulus Selection", True)
        #     mainapp.set_parameter("Processing;Skeleton Analysis", False)
        #     mainapp.set_parameter("Processing;Texture Analysis", False)
        #     if i == 0:
        #         mainapp.set_parameter("Processing;Scan Segmentation;Clean Before Training", True)
        #     else:
        #         mainapp.set_parameter("Processing;Scan Segmentation;Clean Before Training", False)
        #     mainapp.set_parameter("Processing;Scan Segmentation;Run Training", True)
        #     mainapp.set_parameter("Processing;Scan Segmentation;Lobulus Number", 0)
        #     # mainapp.start_gui(qapp=qapp)
        #     mainapp.run_lobuluses()

        assert Path(mainapp.slide_segmentation.clf_fn).exists()
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        modtime1 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        logger.debug(f"classificator prior modification time: {modtime1}")
        assert modtime0 != modtime1
