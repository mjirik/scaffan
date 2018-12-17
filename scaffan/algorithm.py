# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging
logger = logging.getLogger(__name__)

import sys
import os.path as op
import PyQt5
# import PyQt5.QtWidgets
# print("start 3")
# from PyQt5.QtWidgets import QApplication, QFileDialog
# print("start 4")
from PyQt5 import QtGui
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree

from . import image
import io3d
import io3d.datasets

class Scaffan:

    def __init__(self):
        params = [
            {"name": "Input", "type": "group", "children": [
                {'name': 'File Path', 'type': 'str'},
                {'name': 'Select', 'type': 'action'},
                {'name': 'Annotation Color', 'type': 'list', 'values': {
                    "None": None,
                    "White": "#FFFFFF",
                    "Black": "#000000",
                    "Red": "#FF0000",
                    "Green": "#00FF00",
                    "Blue": "#0000FF",
                    "Cyan": "#00FFFF",
                    "Magenta": "#FF00FF",
                    "Yellow": "#FFFF00"},
                 'value': 0},
                # {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
                # {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
            ], },
            {"name": "Output", "type": "group", "children": [
                {'name': 'Directory Path', 'type': 'str', 'value': self._prepare_default_output_dir()},
                {'name': 'Select', 'type': 'action'},
            ], },
            {"name": "Processing", "type": "group", "children": [
                # {'name': 'Directory Path', 'type': 'str', 'value': prepare_default_output_dir()},
                {'name': 'Run', 'type': 'action'},
            ], }
        ]
        self.parameters = Parameter.create(name='params', type='group', children=params)
        self.anim = None
        pass

    def load(self, path):
        self.anim = image.AnnotatedImage(path)

    def run_lobulus(self):
        pass

    def select_file_gui(self):
        from PyQt5 import QtWidgets
        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")

        fn, mask = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Input File", directory=default_dir,
            filter="NanoZoomer Digital Pathology Image(*.ndpi)"
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

        fn = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select Output Directory", directory=default_dir,
            # filter="NanoZoomer Digital Pathology Image(*.ndpi)"
        )
        # print (fn)
        self.set_output_dir(fn)


    def _prepare_default_output_dir(self):
        default_dir = io3d.datasets.join_path(get_root=True)
        # default_dir = op.expanduser("~/data")
        if not op.exists(default_dir):
            default_dir = op.expanduser("~")
        return default_dir

    def run_lobuluses(self):
        fnparam = self.parameters.param("Input", "File Path")
        from .image import AnnotatedImage
        path = self.parameters.param("Input", "File Path")
        anim = AnnotatedImage(path.value())
        print(anim.colors)
        annotation_ids = anim.select_annotations_by_color(list(anim.colors.keys())[0])
        logger.debug("Annotation IDs: {}".format(annotation_ids))
        # print("ann ids", annotation_ids)


    def start_gui(self):

        from PyQt5 import QtWidgets
        # import QApplication, QFileDialog
        app = QtWidgets.QApplication(sys.argv)


        self.parameters.param('Input', 'Select').sigActivated.connect(self.select_file_gui)
        self.parameters.param('Output', 'Select').sigActivated.connect(self.select_output_dir_gui)
        self.parameters.param('Processing', 'Run').sigActivated.connect(self.run_lobuluses)


        t = ParameterTree()
        t.setParameters(self.parameters, showTop=False)
        t.setWindowTitle('pyqtgraph example: Parameter Tree')
        # t.show()


        print("run scaffan")
        win = QtGui.QWidget()
        layout = QtGui.QGridLayout()
        win.setLayout(layout)
        # layout.addWidget(QtGui.QLabel("These are two views of the same data. They should always display the same values."), 0,  0, 1, 2)
        layout.addWidget(t, 1, 0, 1, 1)
        # layout.addWidget(t2, 1, 1, 1, 1)
        win.show()
        win.resize(800, 800)

        app.exec_()

