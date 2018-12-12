# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging
logger = logging.getLogger(__name__)
# print("start")
# from . import image
import os.path as op
import sys
# print("start 2")
import PyQt5
# import PyQt5.QtWidgets
# print("start 3")
# from PyQt5.QtWidgets import QApplication, QFileDialog
# print("start 4")
from PyQt5 import QtGui

# print("start 5")
import io3d
# print("start 6")
import io3d.datasets

from . import image

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
print("Running __main__.py")


def select_file():
    default_dir = io3d.datasets.join_path(get_root=True)
    # default_dir = op.expanduser("~/data")
    if not op.exists(default_dir):
        default_dir = op.expanduser("~")

    fn, mask = QtWidgets.QFileDialog.getOpenFileName(
        None, "Select Input File", directory=default_dir,
        filter="NanoZoomer Digital Pathology Image(*.ndpi)"
    )
    print (fn)
    fnparam = p.param("Input", "File Path")
    fnparam.setValue(fn)
    # import pdb; pdb.set_trace()
    # print("ahoj")


def select_output_dir():
    default_dir = prepare_default_output_dir()

    fn = QtWidgets.QFileDialog.getExistingDirectory(
        None, "Select Output Directory", directory=default_dir,
        # filter="NanoZoomer Digital Pathology Image(*.ndpi)"
    )
    print (fn)
    fnparam = p.param("Output", "Directory Path")
    fnparam.setValue(fn)


def prepare_default_output_dir():
    default_dir = io3d.datasets.join_path(get_root=True)
    # default_dir = op.expanduser("~/data")
    if not op.exists(default_dir):
        default_dir = op.expanduser("~")
    return default_dir

def run():
    fnparam = p.param("Input", "File Path")
    from .image import AnnotatedImage
    path = p.param("Input", "File Path")

    print(path, type(path))
    print(path.getValue())
    anim = AnnotatedImage(str(path))



params = [
    {"name": "Input", "type": "group", "children": [
        {'name': 'File Path', 'type': 'str'},
        {'name': 'Select', 'type': 'action'},
    ], },
    {"name": "Output", "type": "group", "children": [
        {'name': 'Directory Path', 'type': 'str', 'value': prepare_default_output_dir()},
        {'name': 'Select', 'type': 'action'},
    ], },
    {"name": "Processing", "type": "group", "children": [
        # {'name': 'Directory Path', 'type': 'str', 'value': prepare_default_output_dir()},
        {'name': 'Run', 'type': 'action'},
    ], }
    # {'name': 'Basic parameter data types', 'type': 'group', 'children': [
    #     {'name': 'Integer', 'type': 'int', 'value': 10},
    #     {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1},
    #     {'name': 'String', 'type': 'str', 'value': "hi"},
    #     {'name': 'List', 'type': 'list', 'values': [1, 2, 3], 'value': 2},
    #     {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3, 3, 3]}, 'value': 2},
    #     {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
    #     {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
    #     {'name': 'Gradient', 'type': 'colormap'},
    #     {'name': 'Subgroup', 'type': 'group', 'children': [
    #         {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
    #         {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
    #     ]},
    #     {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
    #     {'name': 'Action Parameter', 'type': 'action'},
    # ]},
    # {'name': 'Numerical Parameter Options', 'type': 'group', 'children': [
    #     {'name': 'Units + SI prefix', 'type': 'float', 'value': 1.2e-6, 'step': 1e-6, 'siPrefix': True, 'suffix': 'V'},
    #     {'name': 'Limits (min=7;max=15)', 'type': 'int', 'value': 11, 'limits': (7, 15), 'default': -6},
    #     {'name': 'DEC stepping', 'type': 'float', 'value': 1.2e6, 'dec': True, 'step': 1, 'siPrefix': True,
    #      'suffix': 'Hz'},
    #
    # ]},
    # {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
    #     {'name': 'Save State', 'type': 'action'},
    #     {'name': 'Restore State', 'type': 'action', 'children': [
    #         {'name': 'Add missing items', 'type': 'bool', 'value': True},
    #         {'name': 'Remove extra items', 'type': 'bool', 'value': True},
    #     ]},
    # ]},
    # {'name': 'Extra Parameter Options', 'type': 'group', 'children': [
    #     {'name': 'Read-only', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'readonly': True},
    #     {'name': 'Renamable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'renamable': True},
    #     {'name': 'Removable', 'type': 'float', 'value': 1.2e6, 'siPrefix': True, 'suffix': 'Hz', 'removable': True},
    # ]},
    # ComplexParameter(name='Custom parameter group (reciprocal values)'),
    # ScalableGroup(name="Expandable Parameter Group", children=[
    #     {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
    #     {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
    # ]),
]


from PyQt5 import QtWidgets
# import QApplication, QFileDialog
app = QtWidgets.QApplication(sys.argv)
p = Parameter.create(name='params', type='group', children=params)


p.param('Input', 'Select').sigActivated.connect(select_file)
p.param('Output', 'Select').sigActivated.connect(select_output_dir)
p.param('Processing', 'Run').sigActivated.connect(run)


t = ParameterTree()
t.setParameters(p, showTop=False)
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