# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging

logger = logging.getLogger(__name__)

import sys
import os.path as op
import datetime
from pathlib import Path
import io3d.misc
import json
import time

# import PyQt5.QtWidgets
# print("start 3")
# from PyQt5.QtWidgets import QApplication, QFileDialog
# print("start 4")
from PyQt5 import QtGui

# print("start 5")
from pyqtgraph.parametertree import Parameter, ParameterTree

# print("start 6")

from scaffan import image
import io3d
import io3d.datasets
import scaffan.lobulus
import scaffan.report
import scaffan.skeleton_analysis
from .report import Report
import scaffan.evaluation


class Scaffan:
    def __init__(self):

        self.report: Report = scaffan.report.Report()
        self.report.level = 10
        import scaffan.texture as satex
        self.glcm_textures = satex.GLCMTextureMeasurement()
        self.lobulus_processing = scaffan.lobulus.Lobulus()
        self.skeleton_analysis = scaffan.skeleton_analysis.SkeletonAnalysis()
        self.evaluation = scaffan.evaluation.Evaluation()

        self.lobulus_processing.set_report(self.report)
        self.glcm_textures.set_report(self.report)
        self.skeleton_analysis.set_report(self.report)
        self.evaluation.report = self.report
        params = [
            {
                "name": "Input",
                "type": "group",
                "children": [
                    {"name": "File Path", "type": "str"},
                    {"name": "Select", "type": "action"},
                    {
                        "name": "Annotation Color",
                        "type": "list",
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
                    {"name": "Data Info", "type": "str"},
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
                    {
                        "name": "Run Skeleton Analysis",
                        "type": "bool",
                        "value": True,
                        # "tip": "Show images",
                    },
                    {
                        "name": "Run Texture Analysis",
                        "type": "bool",
                        "value": True,
                        # "tip": "Show images",
                    },
                    self.lobulus_processing.parameters,
                    self.skeleton_analysis.parameters,
                    self.glcm_textures.parameters,
                    {"name": "Run", "type": "action"},
                ],
            },
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
            filter="NanoZoomer Digital Pathology Image(*.ndpi)",
        )
        self.set_input_file(fn)

    def set_input_file(self, fn):
        fnparam = self.parameters.param("Input", "File Path")
        fnparam.setValue(fn)
        # import pdb; pdb.set_trace()
        # print("ahoj")

    def set_output_dir(self, path):
        fnparam = self.parameters.param("Output", "Directory Path")
        fnparam.setValue(path)

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

    def _prepare_default_output_dir(self):
        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        # timestamp = datetime.datetime.now().strftime("SA_%Y-%m-%d_%H:%M:%S")
        timestamp = datetime.datetime.now().strftime("SA_%Y%m%d_%H%M%S")
        default_dir = op.join(default_dir, timestamp)
        return default_dir

    def init_run(self):
        fnparam = self.parameters.param("Input", "File Path")
        path = fnparam.value()
        self.anim = image.AnnotatedImage(path)
        fnparam = self.parameters.param("Output", "Directory Path")
        self.report.set_output_dir(fnparam.value())

    def set_annotation_color_selection(self, color):
        pcolor = self.parameters.param("Input", "Annotation Color")
        color = color.upper()
        if color in pcolor.reverse[0]:
            # val = pcolor.reverse[0].index(color)
            # pcolor.setValue(val)
            pcolor.setValue(color)
        else:
            raise ValueError("Color '{}' not found in allowed colors.".format(color))

    def run_lobuluses(self, event=None):
        self.init_run()
        # if color is None:
        pcolor = self.parameters.param("Input", "Annotation Color")
        # print("color ", pcolor.value())
        # color = pcolor.reverse[0][pcolor.value()]
        color = pcolor.value()
        # print("Color ", color)
        # fnparam = self.parameters.param("Input", "File Path")
        # from .image import AnnotatedImage
        # path = self.parameters.param("Input", "File Path")
        # anim = AnnotatedImage(path.value())
        # if color is None:
        #     color = list(self.anim.colors.keys())[0]
        # print(self.anim.colors)
        annotation_ids = self.anim.select_annotations_by_color(color)
        logger.debug("Annotation IDs: {}".format(annotation_ids))
        # if annotation_ids is None:
        #     logger.error("No color selected")

        show = self.parameters.param("Processing", "Show").value()
        self.report.set_show(show)
        self.report.set_save(True)
        for id in annotation_ids:
            self._run_lobulus(id)


        self.report.df.to_excel(op.join(self.report.outputdir, "data.xlsx"))
        saved_params = self.parameters.saveState()
        io3d.misc.obj_to_file(
            saved_params,
            str(Path(self.report.outputdir) / "parameters.yaml")
        )
        with open(Path(self.report.outputdir) / "parameters.json", "w") as outfile:
            json.dump(saved_params, outfile)
        from . import os_interaction

        open_dir = self.parameters.param("Processing", "Open output dir").value()
        if open_dir:
            os_interaction.open_path(self.report.outputdir)

        # print("ann ids", annotation_ids)

    def _run_lobulus(self, annotation_id):
        show = self.parameters.param("Processing", "Show").value()
        t0 = time.time()

        self.lobulus_processing.set_annotated_image_and_id(self.anim, annotation_id)
        self.lobulus_processing.run(show=show)
        self.skeleton_analysis.set_lobulus(lobulus=self.lobulus_processing)
        if self.parameters.param("Processing", "Run Skeleton Analysis").value():
            self.skeleton_analysis.skeleton_analysis(show=show)
        if self.parameters.param("Processing", "Run Texture Analysis").value():
            # self.glcm_textures.report = self.report
            self.glcm_textures.set_input_data(view=self.lobulus_processing.view, id=annotation_id,
                                              lobulus_segmentation=self.lobulus_processing.lobulus_mask)
            self.glcm_textures.run()
        t1 = time.time()
        self.report.add_cols_to_actual_row({"Processing time [s]": t1 - t0})
        # evaluation
        self.evaluation.set_input_data(self.anim, annotation_id, self.lobulus_processing)
        self.evaluation.run()
        self.report.finish_actual_row()

    def _get_file_info(self):
        fnparam = Path(self.parameters.param("Input", "File Path").value())
        if fnparam.exists():
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
        self.parameters.param("Processing", "Run").sigActivated.connect(
            self.run_lobuluses
        )

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
        # win.
        if not skip_exec:

            qapp.exec_()
