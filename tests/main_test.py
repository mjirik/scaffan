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
from unittest.mock import patch

path_to_script = op.dirname(op.abspath(__file__))
# sys.path.insert(0, op.abspath(op.join(path_to_script, "../../exsu")))
# # sys.path.insert(0, op.abspath(op.join(path_to_script, "../../imma")))
# exsu_pth = Path(__file__).parents[2] / "exsu"
# logger.debug(f"exsupth{exsu_pth}, {exsu_pth.exists()}")
# sys.path.insert(0, exsu_pth)

import exsu

logger.debug(f"exsu path: {exsu.__file__}")
# import openslide
import scaffan
import scaffan.algorithm

# import scaffan
import scaffan.image
from PyQt5 import QtWidgets
import pytest
from datetime import datetime

qapp = QtWidgets.QApplication(sys.argv)


class MainGuiTest(unittest.TestCase):

    # skip_on_local = True
    skip_on_local = False

    # @unittest.skipIf(os.environ.get("TRAVIS", skip_on_local), "Skip on Travis-CI")
    @unittest.skip("Skip interactivet test on Travis-CI")
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
        # mainapp.set_parameter("Input;Lobulus Selection Method", "Color")
        # mainapp.set_parameter("Input;Lobulus Selection Method", 1)
        mainapp.set_annotation_color_selection(
            "#FFFF00", override_automatic_lobulus_selection=True
        )
        mainapp.set_parameter("Processing;Skeleton Analysis", False)
        mainapp.set_parameter("Processing;Texture Analysis", False)
        mainapp.set_parameter("Processing;Scan Segmentation;HCTFS;Run Training", True)
        mainapp.set_parameter("Processing;Scan Segmentation;Lobulus Number", 3)

        # mainapp.set_parameter("Input;Lobulus Selection Method", "Manual")

        lsv = mainapp.get_parameter("Input;Lobulus Selection Method")
        ls = mainapp.get_parameter("Input;Lobulus Selection Method", return_value=False)
        logger.debug(f"lobulus selection={ls}")
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

    # skip_on_local = True

    def test_iteration_limited_snakes_and_texture_analysis(
        self, error_threshold=0.05, iterations1=10, iterations2=10, view_border=20
    ):
        """
        Test multiple lobules processing. Limit snake iterations and the size of view to increase performace.
        :param error_threshold:
        :param iterations1:
        :param iterations2:
        :return:
        """
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )

        logger.debug("in with statement")
        mainapp = scaffan.algorithm.Scaffan(whole_scan_margin=-0.2)
        mainapp.set_input_file(fn)
        mainapp.set_output_dir("test_iteration_limited_snake_and_texture_analysis")
        mainapp.set_annotation_color_selection(
            "#00FFFF", override_automatic_lobulus_selection=True
        )
        # auto = mainapp.get_parameter("Input;Lobulus Selection Method") == "Auto"
        # logger.debug(f"auto={auto}")
        if iterations1:
            mainapp.set_parameter(
                "Processing;Lobulus Segmentation;Border Segmentation;Iterations",
                iterations1,
            )
        if iterations2:
            mainapp.set_parameter(
                "Processing;Lobulus Segmentation;Central Vein Segmentation;Iterations",
                iterations2,
            )
        mainapp.set_parameter(
            "Processing;Lobulus Segmentation;Manual Segmentation", False
        )
        # dont waste time with scan segmentation. It is not used in the test
        mainapp.set_parameter(
            "Processing;Lobulus Segmentation;Annotation Margin", view_border
        )  # add 20%
        mainapp.set_parameter("Processing;Scan Segmentation", False)
        mainapp.set_parameter("Processing;Skeleton Analysis", False)
        mainapp.set_parameter("Processing;Texture Analysis", False)
        original_foo = scaffan.image.AnnotatedImage.get_annotations_by_color
        # with patch.object(scaffan.image.AnnotatedImage, 'select_annotations_by_color', autospec=True) as mock_foo:
        #     def side_effect(anim_, annid, *args, **kwargs):
        #         logger.debug("mocked function select_annotations_by_color()")
        #         original_list = original_foo(anim_, annid, *args, **kwargs)
        #         logger.debug(f"id={annid}, original ann_ids={original_list}")
        #         if annid == "#000000":
        #             new_list = original_list
        #         else:
        #             new_list = original_list[:1]
        #         logger.debug(f"new ann_ids={new_list}")
        #         return new_list
        #
        #     mock_foo.side_effect = side_effect
        mainapp.run_lobuluses()
        self.assert_dice_in_first_evaluated_data(mainapp, error_threshold, i=0)
        self.assert_dice_in_first_evaluated_data(mainapp, error_threshold, i=1)

    def test_run_lobuluses_manual_segmentation_with_texture_and_skeleton_analysis(self, error_threshold=0.9):
        """
        Just check the manual segmentation
        :param error_threshold:
        :return:
        """
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        # original_foo = scaffan.image.AnnotatedImage.get_annotations_by_color
        # with patch.object(scaffan.image.AnnotatedImage, 'select_annotations_by_color', autospec=True) as mock_foo:
        #     def side_effect(anim_, annid, *args, **kwargs):
        #         logger.debug("mocked function select_annotations_by_color()")
        #         original_list = original_foo(anim_, annid, *args, **kwargs)
        #         logger.debug(f"id={annid}, original ann_ids={original_list}")
        #         if annid == "#000000":
        #             new_list = original_list
        #         else:
        #             new_list = original_list[:0]
        #         logger.debug(f"new ann_ids={new_list}")
        #         return new_list
        #     logger.debug("in with statement")
        #     mock_foo.side_effect = side_effect

        mainapp = scaffan.algorithm.Scaffan(whole_scan_margin=-0.2)
        mainapp.set_input_file(fn)
        mainapp.set_output_dir("test_run_lobuluses_output_dir")
        mainapp.set_annotation_color_selection(
            "#FFFF00", override_automatic_lobulus_selection=True
        )
        # auto = mainapp.get_parameter("Input;Lobulus Selection Method") == "Color"
        # logger.debug(f"auto={auto}")
        # if iterations1:
        #     mainapp.set_parameter("Processing;Lobulus Segmentation;Border Segmentation;Iterations", iterations1)
        # if iterations2:
        #     mainapp.set_parameter("Processing;Lobulus Segmentation;Central Vein Segmentation;Iterations", iterations2)
        # Use manual annotations
        mainapp.set_parameter(
            "Processing;Lobulus Segmentation;Manual Segmentation", True
        )
        # dont waste time with scan segmentation. It is not used in the test
        mainapp.set_parameter("Processing;Scan Segmentation", False)
        mainapp.set_parameter("Processing;Skeleton Analysis", True)
        mainapp.set_parameter("Processing;Texture Analysis", True)

        mainapp.run_lobuluses()
        self.assert_dice_in_first_evaluated_data(mainapp, error_threshold, i=0)
        assert mainapp.evaluation.evaluation_history[0]

    def assert_dice_in_first_evaluated_data(self, mainapp, error_threshold, i=0):
        self.assertLess(
            error_threshold,
            mainapp.evaluation.evaluation_history[i]["Lobulus Border Dice"],
            "Lobulus segmentation should have Dice coefficient above some low level",
        )
        self.assertLess(
            error_threshold,
            mainapp.evaluation.evaluation_history[i]["Central Vein Dice"],
            "Central Vein segmentation should have Dice coefficient above some low level",
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

    # @pytest.mark.slow
    def test_training_small_slide_segmentation_clf(self):
        """
        Do the training. Use just one training data and use just every n-th pixel for training.
        :return:
        """

        fns = [
            io3d.datasets.join_path(
                "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
            ),
        ]
        self._slide_segmentation_train_clf(fns, clf_fn=".temp_clf.pkl", stride=1000)

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
        self._testing_slide_segmentation_clf(fns, "HCTFS")

    def test_testing_slide_segmentation_clf(self):
        fns = [
            io3d.datasets.join_path(
                "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
            ),
        ]
        self._testing_slide_segmentation_clf(fns, segmentation_method="HCTFS")

    def test_testing_slide_segmentation_clf_unet(self):
        fns = [
            io3d.datasets.join_path(
                "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
            ),
        ]

        self._testing_slide_segmentation_clf(fns, segmentation_method="U-Net", whole_scan_margin=-0.3)

    def _testing_slide_segmentation_clf(self, fns, segmentation_method, whole_scan_margin=0.0):
        """
        Run whole slide segmentation on all input files and check whether all three labels are
        represented in the output labeling.

        :param fns:
        :param segmentation_method:
        :return:
        """

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
            mainapp.set_parameter("Input;Lobulus Selection Method", "Auto")
            mainapp.set_parameter("Processing;Skeleton Analysis", False)
            mainapp.set_parameter("Processing;Texture Analysis", False)
            mainapp.set_parameter("Processing;Open output dir", False)
            mainapp.set_parameter(
                "Processing;Scan Segmentation;HCTFS;Clean Before Training", False
            )
            mainapp.set_parameter(
                "Processing;Scan Segmentation;Segmentation Method", segmentation_method
            )
            mainapp.set_parameter(
                "Processing;Scan Segmentation;HCTFS;Run Training", False
            )
            # Set some Unet parameter here. It is used if the U-Net Segmentation method is used.
            # mainapp.set_parameter("Processing;Scan Segmentation;U-Net;Some Parameter", False)
            mainapp.set_parameter("Processing;Scan Segmentation;Lobulus Number", 0)
            mainapp.set_parameter("Processing;Whole Scan Margin", whole_scan_margin)
            # mainapp.start_gui(qapp=qapp)
            mainapp.run_lobuluses()

            specimen_size_mm = (
                mainapp.slide_segmentation.sinusoidal_area_mm
                + mainapp.slide_segmentation.septum_area_mm
            )
            whole_area_mm = mainapp.slide_segmentation.empty_area_mm + specimen_size_mm
            logger.debug("asserts")
            assert (
                specimen_size_mm < whole_area_mm
            ), "Specimen should be smaller then whole slide"
            assert specimen_size_mm > whole_area_mm * 0.1, "Specimen should big enough"
            assert (
                mainapp.slide_segmentation.sinusoidal_area_mm > 0.1 * specimen_size_mm
            ), "sinusoidal area should be at least 10% of the specimen area"
            assert (
                mainapp.slide_segmentation.septum_area_mm > 0.1 * specimen_size_mm
            ), "Septum area should be at least 10% of the specimen area"

        assert Path(
            mainapp.slide_segmentation.clf_fn
        ).exists(), "The file with pretrained classifier should exist"
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        modtime1 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        logger.debug(f"classificator prior modification time: {modtime1}")
        assert (
            modtime0 == modtime1
        ), "We are not changing the pretrained classifier file"

    def _slide_segmentation_train_clf(self, fns, clf_fn=None, stride=None):
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
        if stride:
            logger.debug(f"setting stride {stride}")
            mainapp.set_parameter("Processing;Scan Segmentation;HCTFS;Training Stride", stride)
        mainapp.train_scan_segmentation(fns)

        # for i, fn in enumerate(fns):
        #     mainapp.set_input_file(fn)
        #     mainapp.set_output_dir()
        #     # There does not have to be set some color
        #     # mainapp.set_annotation_color_selection("#FF00FF")
        #     # mainapp.set_annotation_color_selection("#FF0000")
        #     mainapp.set_annotation_color_selection("#FFFF00")
        #     mainapp.set_parameter("Input;Lobulus Selection Method", "Auto")
        #     mainapp.set_parameter("Processing;Skeleton Analysis", False)
        #     mainapp.set_parameter("Processing;Texture Analysis", False)
        #     if i == 0:
        #     else:
        #         mainapp.set_parameter("Processing;Scan Segmentation;HCTFS;Clean Before Training", False)
        #     mainapp.set_parameter("Processing;Scan Segmentation;HCTFS;Run Training", True)
        #     mainapp.set_parameter("Processing;Scan Segmentation;Lobulus Number", 0)
        #     # mainapp.start_gui(qapp=qapp)
        #     mainapp.run_lobuluses()

        assert Path(mainapp.slide_segmentation.clf_fn).exists()
        clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        modtime1 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        logger.debug(f"classificator prior modification time: {modtime1}")
        assert modtime0 != modtime1

    def test_testing_slide_segmentation_clf_unet_controled_parameters(self):
        fns = [
            io3d.datasets.join_path(
                "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
            ),
        ]

        # TODO Uncomment fallowing line when CNN is done
        # self._testing_slide_segmentation_clf(fns, segmentation_method="U-Net")

        mainapp = scaffan.algorithm.Scaffan()
        # clf_fn = mainapp.slide_segmentation.clf_fn
        # assert clf_fn.exists()
        #
        # if clf_fn.exists():
        #     modtime0 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        # else:
        #     modtime0 = ""
        # logger.debug(f"classificator prior modification time: {modtime0}")
        #
        # for fn in fns:
        #     mainapp.set_input_file(fn)
        #     mainapp.set_output_dir()
        #     # There does not have to be set some color
        #     # mainapp.set_annotation_color_selection("#FF00FF")
        #     # mainapp.set_annotation_color_selection("#FF0000")
        #     mainapp.set_annotation_color_selection("#FFFF00")
        #     mainapp.set_parameter("Input;Lobulus Selection Method", "Auto")
        #     mainapp.set_parameter("Processing;Skeleton Analysis", False)
        #     mainapp.set_parameter("Processing;Texture Analysis", False)
        #     mainapp.set_parameter("Processing;Open output dir", False)
        #     mainapp.set_parameter(
        #         "Processing;Scan Segmentation;HCTFS;Clean Before Training", False
        #     )
        #     mainapp.set_parameter("Processing;Scan Segmentation;Segmentation Method", "U-Net")
        #     mainapp.set_parameter("Processing;Scan Segmentation;Working Tile Size", 224)
        #     mainapp.set_parameter("Processing;Scan Segmentation;HCTFS;Run Training", False)
        #     # Set some Unet parameter here. It is used if the U-Net Segmentation method is used.
        #     # mainapp.set_parameter("Processing;Scan Segmentation;U-Net;Some Parameter", False)
        #     mainapp.set_parameter("Processing;Scan Segmentation;Lobulus Number", 0)
        #     # mainapp.start_gui(qapp=qapp)
        #     mainapp.run_lobuluses()
        #
        #     specimen_size_mm = (
        #             mainapp.slide_segmentation.sinusoidal_area_mm
        #             + mainapp.slide_segmentation.septum_area_mm
        #     )
        #     whole_area_mm = mainapp.slide_segmentation.empty_area_mm + specimen_size_mm
        #     logger.debug("asserts")
        #     assert specimen_size_mm < whole_area_mm, "Specimen should be smaller then whole slide"
        #     assert specimen_size_mm > whole_area_mm * 0.1, "Specimen should big enough"
        #     assert (
        #             mainapp.slide_segmentation.sinusoidal_area_mm > 0.1 * specimen_size_mm
        #     ), "sinusoidal area should be at least 10% of the specimen area"
        #     assert mainapp.slide_segmentation.septum_area_mm > 0.1 * specimen_size_mm, \
        #         "Septum area should be at least 10% of the specimen area"
        #
        # assert Path(mainapp.slide_segmentation.clf_fn).exists(), "The file with pretrained classifier should exist"
        # clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        # modtime1 = datetime.fromtimestamp(clf_fn.stat().st_mtime)
        # logger.debug(f"classificator prior modification time: {modtime1}")
        # assert modtime0 == modtime1, "We are not changing the pretrained classifier file"


# @pytest.mark.parametrize("fn_yellow")

def test_parameters():
    fn_cssf= "pokus.xlsx"
    mainapp = scaffan.algorithm.Scaffan()
    fn_cssf_orig = mainapp.get_parameter("Output;Common Spreadsheet File")
    mainapp.set_common_spreadsheet_file(fn_cssf)
    fn_csfn_read = mainapp.get_parameter("Output;Common Spreadsheet File")
    mainapp.set_common_spreadsheet_file(fn_cssf_orig)
    assert fn_csfn_read == fn_cssf




# @unittest.skipIf(os.environ.get("TRAVIS", True), "Skip on Travis-CI")
@pytest.mark.slow
def test_run_lobuluses():
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    mainapp = scaffan.algorithm.Scaffan()
    mainapp.set_parameter("Processing;Scan Segmentation", False)
    run_on_yellow(mainapp, fn)


# @unittest.skip("Skip it is somehow broken")
@pytest.mark.slow
def test_run_lobuluses_czi():
    # TODO fix test
    fn = io3d.datasets.join_path(
        "medical/orig/scaffan-analysis-czi/Zeiss-scans/05_2019_11_12__-1-2.czi",
        get_root=True,
    )
    mainapp = scaffan.algorithm.Scaffan()
    mainapp.set_parameter("Processing;Scan Segmentation", False)
    run_on_yellow(mainapp, fn)


def run_on_yellow(mainapp, fn_yellow):
    # imsl = openslide.OpenSlide(fn)
    # annotations = scan.read_annotations(fn)
    # scan.annotations_to_px(imsl, annotations)
    mainapp.set_input_file(fn_yellow)
    mainapp.set_output_dir("test_run_lobuluses_output_dir")
    # mainapp.init_run()
    # mainapp.set_annotation_color_selection("#FF00FF") # magenta -> cyan
    # mainapp.set_annotation_color_selection("#00FFFF")
    # cyan causes memory fail
    mainapp.set_parameter("Input;Lobulus Selection Method", "Color")
    mainapp.set_annotation_color_selection("#FFFF00")
    mainapp.run_lobuluses()
    assert (
        0.6 < mainapp.evaluation.evaluation_history[0]["Lobulus Border Dice"]
    ), "Lobulus segmentation should have Dice coefficient above some low level"
    # self.assertLess(0.6, mainapp.evaluation.evaluation_history[1]["Lobulus Border Dice"],
    #                 "Lobulus segmentation should have Dice coefficient above some low level")
    assert (
        0.2 < mainapp.evaluation.evaluation_history[0]["Central Vein Dice"]
    ), "Central Vein segmentation should have Dice coefficient above some low level"
    # self.assertLess(0.5, mainapp.evaluation.evaluation_history[1]["Central Vein Dice"],
    #                 "Central Vein should have Dice coefficient above some low level")
