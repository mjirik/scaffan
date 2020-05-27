# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""

from loguru import logger

from scaffan import image
import sys
import os.path as op
import datetime
from pathlib import Path
import io3d.misc
from io3d import cachefile
import json
import time
import platform
from typing import List, Union

# import PyQt5.QtWidgets
# print("start 3")
# from PyQt5.QtWidgets import QApplication, QFileDialog
# print("start 4")
from PyQt5 import QtGui

# print("start 5")
from pyqtgraph.parametertree import Parameter, ParameterTree
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use('Qt5Agg')

# print("start 6")

import io3d
import io3d.datasets
import scaffan.lobulus
import scaffan.skeleton_analysis
from exsu.report import Report
import scaffan.evaluation
import scaffan.slide_segmentation
from scaffan.pyqt_widgets import BatchFileProcessingParameter
from scaffan.image_intensity_rescale_pyqtgraph import RescaleIntensityPercentilePQG
from . import sni_prediction


class Scaffan:
    def __init__(
            self,
            whole_scan_margin=-0.0
    ):
        """

        :param whole_scan_margin: negative value makes the the area for automatic lobuli selection smaller. It is used in
        tests and debugging.
        """

        self.report: Report = Report(
            repodir=Path(__file__).parent.resolve(),
            check_version_of=["numpy", "scipy", "skimage"],
        )
        # self.report.level = 50

        self.raise_exception_if_color_not_found = True

        import scaffan.texture as satex

        sni_predictor = sni_prediction.SniPredictor(
            report=self.report, ptype="bool", pvalue=True
        )
        self.glcm_textures = satex.GLCMTextureMeasurement(
            report=self.report, sni_predictor=sni_predictor
        )
        self.lobulus_processing = scaffan.lobulus.Lobulus(
            ptype="bool", report=self.report
        )
        self.whole_scan_margin = whole_scan_margin
        self.skeleton_analysis = scaffan.skeleton_analysis.SkeletonAnalysis()
        self.evaluation = scaffan.evaluation.Evaluation()
        self.intensity_rescale = RescaleIntensityPercentilePQG()
        self.slide_segmentation = scaffan.slide_segmentation.ScanSegmentation(report=self.report)
        # self.slide_segmentation.report = self.report

        # self.lobulus_processing.set_report(self.report)
        # self.glcm_textures.set_report(self.report)
        self.skeleton_analysis.set_report(self.report)
        self.evaluation.report = self.report
        self.win: QtGui.QWidget = None
        self.cache = cachefile.CacheFile("~/.scaffan_cache.yaml")
        # self.cache.update('', path)
        common_spreadsheet_file = self.cache.get_or_save_default(
            "common_spreadsheet_file",
            self._prepare_default_output_common_spreadsheet_file(),
        )
        logger.debug(
            "common_spreadsheet_file loaded as: {}".format(common_spreadsheet_file)
        )
        params = [
            {
                "name": "Input",
                "type": "group",
                "children": [
                    {"name": "File Path", "type": "str"},
                    {"name": "Select", "type": "action"},
                    {"name": "Data Info", "type": "str", "readonly": True},
                    {
                        "name": "Lobulus Selection Method",
                        "type": "list",
                        "value": "Auto",

                        "values": ["Color","Manual", "Auto"],
                            # "Color": "Color",
                            # "Manual": "Manual",
                            # "Auto": "Auto",
                        # },
                        "tip": "Auto: select lobulus based on Scan Segmentation.\n"
                               "Color: based on annotation color.\n"
                               "Manual: manually pick the lobule. \n\n" +
                       "The annotations from `.ndpi` files are automatically imported.\n " +
                       "The annotattion for other image format are expected to be done in ImageJ ROI Manager.\n"
                       "File with annotations for is expected to be in the same dir and same name "
                       "with file extension `.roi.zip`. \n"
                       "Color of segmentation can be part of the polyon name ('my annotation #00FF00')"
                    },
                    # {
                    #     "name": "Lobulus Selection Method",
                    #     "type": "bool",
                    #     "value": True,
                    #     "tip": "Skip selection based on annotation color and select lobulus based on Scan Segmentation. ",
                    # },
                    {
                        "name": "Annotation Color",
                        "type": "list",
                        "tip": "Select lobulus based on annotation color. "
                               "Color of the annotation from ImageJ is expected to be part of the polyon name ('my annotation #00FF00')"
                               "Skipped if Automatic Lobulus Selection Method is used.",
                        "values": {
                            "None": None,
                            "White": "#FFFFFF",
                            "Black": "#000000",
                            "Red": "#FF0000",
                            "Green": "#00FF00",
                            "Blue": "#0000FF",
                            "Cyan": "#00FFFF",
                            "Magenta": "#FF00FF",
                            "Yellow": "#FFFF00",
                        },
                        "value": 0,
                    },
                    # {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
                    # {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
                    # BatchFileProcessingParameter(
                    #     name="Batch processing", children=[]
                    # ),
                    # {
                    #     "name": "Advanced",
                    #     "type": "group",
                    #     "children": [
                    #         dict(name="Ignore not found color",type="bool", value=False,
                    #              tip="No exception thrown if color not found in the data"),
                    #     ]
                    # }
                ],
            },
            {
                "name": "Output",
                "type": "group",
                "children": [
                    {
                        "name": "Directory Path",
                        "type": "str",
                        "value": self._prepare_default_output_dir(),
                    },
                    {"name": "Select", "type": "action"},
                    {
                        "name": "Common Spreadsheet File",
                        "type": "str",
                        "value": common_spreadsheet_file,
                    },
                    {
                        "name": "Select Common Spreadsheet File",
                        "type": "action",
                        "tip": "All measurements are appended to this file in addition to data stored in Output Directory Path.",
                    },
                ],
            },
            {
                "name": "Processing",
                "type": "group",
                "children": [
                    # {'name': 'Directory Path', 'type': 'str', 'value': prepare_default_output_dir()},
                    {
                        "name": "Show",
                        "type": "bool",
                        "value": False,
                        "tip": "Show images",
                    },
                    {
                        "name": "Open output dir",
                        "type": "bool",
                        "value": False,
                        "tip": "Open system window with output dir when processing is finished",
                    },
                    # {
                    #     "name": "Run Scan Segmentation",
                    #     "type": "bool",
                    #     "value": True,
                    #     "tip": "Run analysis of whole slide before all other processing is perfomed",
                    # },
                    # {
                    #     "name": "Skeleton Analysis",
                    #     "type": "bool",
                    #     "value": True,
                    #     # "tip": "Show images",
                    # },
                    # {
                    #     "name": "Texture Analysis",
                    #     "type": "bool",
                    #     "value": True,
                    #     # "tip": "Show images",
                    # },
                    self.intensity_rescale.parameters,
                    self.slide_segmentation.parameters,
                    self.lobulus_processing.parameters,
                    self.skeleton_analysis.parameters,
                    self.glcm_textures.parameters,
                    {
                        "name": "Whole Scan Margin",
                        "type": "float",
                        "value": self.whole_scan_margin,
                        "tip": "Negative value will crop the whole scan image",
                    },
                    {
                        "name": "Preview Pixelsize",
                        "type": "float",
                        "value": 0.00002,  # 0.02 mm
                        "suffix": "m",
                        "siPrefix": True,
                        "tip": "Pixelsize used for manual lobuli selection. ",
                    },
                    {
                        "name": "Report Level",
                        "type": "int",
                        "value": 50,
                        "tip": "Control ammount of stored images. 0 - all debug imagess will be stored. "
                        "100 - just important images will be saved.",
                    },
                ],
            },
            {"name": "Run", "type": "action"},
        ]
        self.parameters = Parameter.create(name="params", type="group", children=params)
        # here is everything what should work with or without GUI
        self.parameters.param("Input", "File Path").sigValueChanged.connect(
            self._get_file_info
        )
        self.anim: image.AnnotatedImage = None
        pass

    def select_file_gui(self):
        from PyQt5 import QtWidgets

        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        fn, mask = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select Input File",
            directory=default_dir,
            filter="Images (*.ndpi *.czi *.tif *.tiff);;NanoZoomer Digital Pathology Image (*.ndpi);;" +
                   "Zeiss Image format for microscopes (*.czi);;Tiff image (*.tiff *.tif)",
        )
        self.set_input_file(fn)

    def set_input_file(self, fn: Union[Path, str]):
        fn = str(fn)
        fnparam = self.parameters.param("Input", "File Path")
        fnparam.setValue(fn)
        logger.debug("Set Input File Path to : {}".format(fn))
        # import pdb; pdb.set_trace()
        # print("ahoj")

    def set_output_dir(self, path: Union[str, Path] = None):
        """
        Set directory for all outputs. The standard
        :param path: if no parameter is given the standard path in ~/data/SA_%Date_%Time is selected
        :return:
        """
        if path is None:
            path = self._prepare_default_output_dir()
        logger.debug(f"output directory path={path}")
        fnparam = self.parameters.param("Output", "Directory Path")
        fnparam.setValue(str(path))

    def set_report_level(self, level: int):
        fnparam = self.parameters.param("Processing", "Report Level")
        fnparam.setValue(level)

    def set_common_spreadsheet_file(self, path):
        fnparam = self.parameters.param("Output", "Common Spreadsheet File")
        fnparam.setValue(path)
        self.cache.update("common_spreadsheet_file", path)
        logger.info("common_spreadsheet_file set to {}".format(path))
        # print("common_spreadsheet_file set to {}".format(path))

    def get_parameter(self, param_path, parse_path=True, return_value=True):
        if parse_path:
            param_path = param_path.split(";")
        fnparam = self.parameters.param(*param_path)
        if return_value:
            return fnparam.value()
        else:
            return fnparam

    def set_parameter(self, param_path, value, parse_path=True):
        """
        Set value to parameter.
        :param param_path: Path to parameter can be separated by ";"
        :param value:
        :param parse_path: Turn on separation of path by ";"
        :return:
        """
        logger.debug(f"Set {param_path} to {value}")
        if parse_path:
            param_path = param_path.split(";")
        fnparam = self.parameters.param(*param_path)
        fnparam.setValue(value)

    def select_output_dir_gui(self):
        from PyQt5 import QtWidgets

        default_dir = self._prepare_default_output_dir()
        if op.exists(default_dir):
            start_dir = default_dir
        else:
            start_dir = op.dirname(default_dir)

        fn = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "Select Output Directory",
            directory=start_dir,
            # filter="NanoZoomer Digital Pathology Image(*.ndpi)"
        )
        # print (fn)
        self.set_output_dir(fn)

    def select_output_spreadsheet_gui(self):
        from PyQt5 import QtWidgets

        default_dir = self._prepare_default_output_dir()
        if op.exists(default_dir):
            start_dir = default_dir
        else:
            start_dir = op.dirname(default_dir)

        fn = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Select Common Spreadsheet File",
            directory=start_dir,
            filter="Excel File (*.xlsx)",
        )[0]
        # print (fn)
        self.set_common_spreadsheet_file(fn)

    def _prepare_default_output_common_spreadsheet_file(self):
        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        # timestamp = datetime.datetime.now().strftime("SA_%Y-%m-%d_%H:%M:%S")
        # timestamp = datetime.datetime.now().strftime("SA_%Y%m%d_%H%M%S")
        default_dir = op.join(default_dir, "SA_data.xlsx")
        return default_dir

    def _prepare_default_output_dir(self):
        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        # timestamp = datetime.datetime.now().strftime("SA_%Y-%m-%d_%H:%M:%S")
        timestamp = datetime.datetime.now().strftime("SA_%Y%m%d_%H%M%S")
        default_dir = op.join(default_dir, timestamp)
        return default_dir

    def parameters_to_dict(self):
        from . import dilipg

        return dilipg.params_and_values(self.parameters)

    def init_run(self):
        logger.debug("Init Run")
        fnparam = self.parameters.param("Input", "File Path")
        path = fnparam.value()
        # run_resc_int = self.parameters.param("Processing", "Intensity Normalization", "Run Intensity Normalization").value()
        self.anim = image.AnnotatedImage(path)
        self.intensity_rescale.set_anim_params(self.anim)
        fnparam = self.parameters.param("Output", "Directory Path")
        self.report.init_with_output_dir(fnparam.value())
        logger.debug(f"report output dir: {self.report.outputdir}")
        fn_spreadsheet = self.parameters.param("Output", "Common Spreadsheet File")
        self.report.additional_spreadsheet_fn = str(fn_spreadsheet.value())

    def set_annotation_color_selection(
        self, color: str, override_automatic_lobulus_selection=True
    ):
        if override_automatic_lobulus_selection:
            logger.debug("forced to use color selection")
            self.set_parameter("Input;Lobulus Selection Method", "Color")
        logger.debug(f"color={color}")
        pcolor = self.parameters.param("Input", "Annotation Color")
        color = color.upper()
        color_name = color.lower()
        color_name = color_name.capitalize()
        color_names = dict(zip(*pcolor.reverse[::-1]))
        if color_name in color_names:
            color = color_names[color_name]

        # rewrite name to code
        if color in pcolor.reverse[0]:
            # val = pcolor.reverse[0].index(color)
            # pcolor.setValue(val)
            logger.debug(f"setting color parameter to {color}")
            pcolor.setValue(color)
        else:
            raise ValueError("Color '{}' not found in allowed colors.".format(color))

    def train_scan_segmentation(
        self, fns: List[Union[str, Path]], clean_before_training=True
    ):
        """
        Train scan segmentation based on list of files with annotation.
        Output dir set before processing is ignored.
        :param fns: list of filenames
        :return:
        """
        #     mainapp = scaffan.algorithm.Scaffan()
        # if clf_fn is not None:
        #     mainapp.slide_segmentation.clf_fn = Path(clf_fn)
        # clf_fn = Path(mainapp.slide_segmentation.clf_fn)
        clf_fn = Path(self.slide_segmentation.clf_fn)
        logger.debug(f"train clf on path: {clf_fn}, exists: {clf_fn.exists()}")

        for i, fn in enumerate(fns):
            self.set_input_file(fn)
            self.set_output_dir()
            # There does not have to be set some color
            # mainapp.set_annotation_color_selection("#FF00FF")
            # mainapp.set_annotation_color_selection("#FF0000")
            #             mainapp.set_annotation_color_selection("#FFFF00")
            self.set_parameter("Input;Lobulus Selection Method", "Auto")
            self.set_parameter("Processing;Skeleton Analysis", False)
            self.set_parameter("Processing;Texture Analysis", False)
            self.set_parameter("Processing;Lobulus Segmentation", False)
            if i == 0:
                if clean_before_training is not None:
                    self.set_parameter(
                        "Processing;Scan Segmentation;HCTFS;Clean Before Training",
                        clean_before_training,
                    )
            else:
                self.set_parameter(
                    "Processing;Scan Segmentation;HCTFS;Clean Before Training", False
                )
            self.set_parameter("Processing;Scan Segmentation;HCTFS;Run Training", True)
            #             mainapp.set_parameter("Processing;Slide Segmentation;Lobulus Number", 0)
            # mainapp.start_gui(qapp=qapp)
            self.run_lobuluses()

    def run_lobuluses(self, event=None):
        self.init_run()
        annotation_ids = None
        # if color is None:
        pcolor = self.parameters.param("Input", "Annotation Color")
        color = pcolor.value()
        self.report.level = self.parameters.param("Processing", "Report Level").value()
        # print("Color ", color)
        # fnparam = self.parameters.param("Input", "File Path")
        # from .image import AnnotatedImage
        # path = self.parameters.param("Input", "File Path")
        # anim = AnnotatedImage(path.value())
        # if color is None:
        #     color = list(self.anim.colors.keys())[0]
        # print(self.anim.colors)
        # if annotation_ids is None:
        #     logger.error("No color selected")

        show = self.parameters.param("Processing", "Show").value()
        self.report.set_show(show)
        self.report.set_save(True)
        run_slide_segmentation = self.parameters.param(
            "Processing", "Scan Segmentation"
        ).value()
        automatic_lobulus_selection = self.parameters.param(
            "Input", "Lobulus Selection Method"
        ).value()
        logger.debug(f"Lobulus Selection={automatic_lobulus_selection}")
        if automatic_lobulus_selection == "Color":
            annotation_ids = self.anim.get_annotations_by_color(
                color,
                raise_exception_if_not_found=self.raise_exception_if_color_not_found,
            )
        # elif automatic_lobulus_selection and not run_slide_segmentation:
        if automatic_lobulus_selection == "Manual":
            annotation_ids = self.manual_select()
        if run_slide_segmentation:
            # fn_input = self.parameters.param("Input", "File Path").value()

            # TODO remove when whole scan will work fine
            # wsm = self.whole_scan_margin
            wsm = self.parameters.param("Processing", "Whole Scan Margin").value()
            self.slide_segmentation.init(self.anim.get_full_view(margin=wsm))
            self.slide_segmentation.run()
            if automatic_lobulus_selection == "Auto":
                self.slide_segmentation.add_biggest_to_annotations()
                annotation_ids = self.slide_segmentation.ann_biggest_ids


        if annotation_ids is None:
            raise NoLobulusSelectionUsedError
        logger.debug("Annotation IDs: {}".format(annotation_ids))
        run_lob = self.parameters.param("Processing", "Lobulus Segmentation").value()
        if run_lob:
            for id in annotation_ids:
                self._run_lobulus(id)

        # in the case no lobulus has been measured the segmentation measurement is stored to table
        if not run_lob or (run_slide_segmentation and len(annotation_ids) == 0):
            self._add_general_information_to_actual_row()
            self.report.finish_actual_row()

        # self.report.df.to_excel(op.join(self.report.outputdir, "data.xlsx"))
        dumped = False
        while not dumped:
            try:
                # logger.debug(f"data frame {self.report.df}")
                logger.debug("Saving dataframe")
                self.report.dump()
                dumped = True
            except PermissionError as e:
                logger.error("Permission Error")
                if self.win is not None:
                    ret = QtGui.QMessageBox.warning(
                        self.win,
                        "XLSX file opened in external application",
                        "Close opened spreadsheet files before continue",
                    )

                else:
                    raise e
        saved_params = self.parameters.saveState()
        io3d.misc.obj_to_file(
            saved_params, str(Path(self.report.outputdir) / "parameters.yaml")
        )
        try:
            with open(
                str(Path(self.report.outputdir) / "parameters.json"), "w"
            ) as outfile:
                json.dump(saved_params, outfile)
        except:
            import traceback

            logger.debug("saved_params: " + str(saved_params))
            logger.warning(f"Problem with dump file to json: {traceback.format_exc()}")
        from . import os_interaction

        open_dir = self.parameters.param("Processing", "Open output dir").value()
        if open_dir:
            os_interaction.open_path(self.report.outputdir)
        logger.debug("finished")

        # print("ann ids", annotation_ids)

    def _add_general_information_to_actual_row(self):
        inpath = Path(self.parameters.param("Input", "File Path").value())
        fn = inpath.parts[-1]
        fn_out = self.parameters.param("Output", "Directory Path").value()
        self.report.add_cols_to_actual_row(
            {
                "File Name": str(fn),
                "File Path": str(inpath),
                "Annotation Color": self.parameters.param(
                    "Input", "Annotation Color"
                ).value(),
                "Datetime": datetime.datetime.now().isoformat(" ", "seconds"),
                "platform.system": platform.uname().system,
                "platform.node": platform.uname().node,
                "platform.processor": platform.uname().processor,
                "Scaffan Version": scaffan.__version__,
                "Output Directory Path": str(fn_out),
            }
        )
        self.report.add_cols_to_actual_row(self.parameters_to_dict())

    def _run_lobulus(self, annotation_id):
        show = self.parameters.param("Processing", "Show").value()
        t0 = time.time()
        inpath = Path(self.parameters.param("Input", "File Path").value())
        fn = inpath.parts[-1]
        self.report.add_cols_to_actual_row(
            {
                # "File Name": str(fn),
                "Annotation ID": annotation_id,
            }
        )
        logger.info(f"Processing file: {fn} with Annotation ID: {annotation_id}")

        self.lobulus_processing.set_annotated_image_and_id(self.anim, annotation_id)
        self.lobulus_processing.run(show=show)
        logger.trace(
            f"type lobulus_processing.lobulus_mask: {type(self.lobulus_processing.lobulus_mask)}"
        )
        self.skeleton_analysis.set_lobulus(lobulus=self.lobulus_processing)
        logger.debug("set lobulus done")
        # run_slide_segmentation = self.parameters.param("Processing", "Texture Analysis").value()
        run_skeleton_analysis = self.parameters.param(
            "Processing", "Skeleton Analysis"
        ).value()
        run_texture_analysis = self.parameters.param(
            "Processing", "Texture Analysis"
        ).value()
        logger.debug("before skeleton analysis")
        if run_skeleton_analysis:
            self.skeleton_analysis.skeleton_analysis(show=show)
        if run_texture_analysis:
            # self.glcm_textures.report = self.report
            self.glcm_textures.set_input_data(
                view=self.lobulus_processing.view,
                annotation_id=annotation_id,
                lobulus_segmentation=self.lobulus_processing.lobulus_mask,
            )
            self.glcm_textures.run()
        logger.trace("after texture analysis")
        t1 = time.time()
        ann_center = self.anim.get_annotation_center_mm(annotation_id)
        # inpath = Path(self.parameters.param("Input", "File Path").value())
        # fn = inpath.parts[-1]
        self._add_general_information_to_actual_row()
        self.report.add_cols_to_actual_row(
            {
                # "File Path": str(inpath),
                # "Annotation Color": self.parameters.param("Input", "Annotation Color").value(),
                "Annotation Center X [mm]": ann_center[0],
                "Annotation Center Y [mm]": ann_center[1],
                "Processing Time [s]": t1 - t0,
                # "Datetime": datetime.datetime.now().isoformat(' ', 'seconds'),
                # "platform.system": platform.uname().system,
                # "platform.node": platform.uname().node,
                # "platform.processor": platform.uname().processor,
                # "Scaffan Version": scaffan.__version__,
            }
        )
        # evaluation
        self.evaluation.set_input_data(
            self.anim, annotation_id, self.lobulus_processing
        )
        self.evaluation.run()
        # Copy all parameters to table
        self.report.finish_actual_row()

    def manual_select(self):
        logger.debug("Manual selection")
        # full_view = self.anim.get_full_view()
        full_view = self.anim.get_view(location=[0,0], level=0, size_on_level=self.anim.get_slide_size()[::-1])
        pxsz_mm = float(self.get_parameter("Processing;Preview Pixelsize")) * 1000
        view_corner = full_view.to_pixelsize(pixelsize_mm=[pxsz_mm, pxsz_mm])
        logger.debug(f"Manual selection1, view.loc={full_view.region_location}, view.size={full_view.region_size_on_level}, pxsz={full_view.region_pixelsize}")
        logger.debug(f"Manual selection2, view.loc={view_corner.region_location}, view.size={view_corner.region_size_on_level}, pxsz={view_corner.region_pixelsize}")
        img = view_corner.get_region_image(as_gray=False)
        plt.ioff()
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.set_window_title("Select lobules. Left/Middle/Right Mouse Button: add/quit/remove")
        logger.debug(f"Manual selection2 backend={matplotlib.get_backend()}, ion={matplotlib.is_interactive()}, img.shape={img.shape}, img.max={np.max(img)}")
        plt.imshow(img)
        # plt.ginput(1)
        # plt.axis("image")
        # fig.ax
        # fig.axes[0].imshow(img)
        # logger.debug("Manual selection4")
        points_px = np.asarray(plt.ginput(-1))
        plt.close(fig)
        logger.debug(f"Manual selection5, centers_px={points_px}")



        # points_px = np.asarray(centers_px)
        from scaffan.image import get_offset_px, get_pixelsize

        # offset_px = get_offset_px(self.anim)
        # pixelsize, pixelunit = get_pixelsize(self.anim, requested_unit="mm")
        x_px_view =  points_px[:,0].flatten()
        y_px_view =  points_px[:,1].flatten()
        # x_px = (x_px_view*view_corner.zoom[0] + offset_px[0])
        # y_px = (y_px_view*view_corner.zoom[1] + offset_px[1])
        # pts_glob_px = [x_px, y_px]

        pts_glob_px = view_corner.coords_view_px_to_glob_px(
            x_px_view, y_px_view
        )
        centers_px = list(zip(*pts_glob_px))
        logger.debug(f"Manual selection5, centers_px_global={centers_px}")
        # centers_px = list(zip(*pts_glob_px))
        r_mm = float(self.get_parameter("Processing;Scan Segmentation;Annotation Radius")) * 1000

        ann_ids, _ = scaffan.slide_segmentation.add_circle_annotation(view_corner, centers_px, annotations=self.anim.annotations, r_mm=r_mm)
        view_corner.set_annotations(self.anim.annotations)
        view_corner.adjust_annotation_to_image_view()
        # logger.debug(f"annotations={self.anim.annotations}")

        logger.debug("annotation selected")
        fig = plt.figure()
        plt.imshow(img)
        view_corner.plot_annotations(None)
        centers_px = plt.ginput(1)
        plt.draw()
        plt.show()
        plt.draw()
        plt.close(fig)
        return ann_ids

    def set_persistent_cols(self, dct: dict):
        """
        Set data which will be appended to all rows.
        :param dct: dictionary with column name and value
        :return:
        """
        self.report.set_persistent_cols(dct)

    def _get_file_info(self):
        fnparam = Path(self.parameters.param("Input", "File Path").value())
        if fnparam.exists() and fnparam.is_file():
            anim = scaffan.image.AnnotatedImage(str(fnparam))
            self.parameters.param("Input", "Data Info").setValue(anim.get_file_info())
            # self.parameters.param("Input", "Select").setValue(anim.get_file_info())

    def start_gui(self, skip_exec=False, qapp=None):

        from PyQt5 import QtWidgets
        import scaffan.qtexceptionhook

        # import QApplication, QFileDialog
        if not skip_exec and qapp == None:
            qapp = QtWidgets.QApplication(sys.argv)

        self.parameters.param("Input", "Select").sigActivated.connect(
            self.select_file_gui
        )
        self.parameters.param("Output", "Select").sigActivated.connect(
            self.select_output_dir_gui
        )
        self.parameters.param(
            "Output", "Select Common Spreadsheet File"
        ).sigActivated.connect(self.select_output_spreadsheet_gui)
        self.parameters.param("Run").sigActivated.connect(self.run_lobuluses)

        self.parameters.param("Processing", "Open output dir").setValue(True)
        t = ParameterTree()
        t.setParameters(self.parameters, showTop=False)
        t.setWindowTitle("pyqtgraph example: Parameter Tree")
        t.show()

        # print("run scaffan")
        win = QtGui.QWidget()
        win.setWindowTitle("ScaffAn {}".format(scaffan.__version__))
        logo_fn = op.join(op.dirname(__file__), "scaffan_icon256.png")
        app_icon = QtGui.QIcon()
        # app_icon.addFile(logo_fn, QtCore.QSize(16, 16))
        app_icon.addFile(logo_fn)
        win.setWindowIcon(app_icon)
        # qapp.setWindowIcon(app_icon)
        layout = QtGui.QGridLayout()
        win.setLayout(layout)
        pic = QtGui.QLabel()
        pic.setPixmap(QtGui.QPixmap(logo_fn).scaled(100, 100))
        pic.show()
        # layout.addWidget(QtGui.QLabel("These are two views of the same data. They should always display the same values."), 0,  0, 1, 2)
        layout.addWidget(pic, 1, 0, 1, 1)
        layout.addWidget(t, 2, 0, 1, 1)
        # layout.addWidget(t2, 1, 1, 1, 1)

        win.show()
        win.resize(800, 800)
        self.win = win
        # win.
        if not skip_exec:

            qapp.exec_()


class NoLobulusSelectionUsedError(Exception):
    pass
