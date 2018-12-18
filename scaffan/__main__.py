# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging
logger = logging.getLogger(__name__)
# print("start")
# from . import image

# print("start 5")
import io3d
# print("start 6")
import io3d.datasets

from scaffan import image
from scaffan import algorithm

print("Running __main__.py")
mainapp = algorithm.Scaffan()
mainapp.start_gui()






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


