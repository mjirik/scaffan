#! /usr/bin/python
# -*- coding: utf-8 -*-

from loguru import logger
import unittest

# import openslide
import scaffan
import scaffan.algorithm
import pyqtgraph.parametertree

import scaffan.dilipg


class ParameterTest(unittest.TestCase):

    skip_on_local = False
    # skip_on_local = True
    # @unittest.skipIf(os.environ.get("TRAVIS", skip_on_local), "Skip on Travis-CI")
    def test_export_to_dict(self):
        mainapp = scaffan.algorithm.Scaffan()
        p = mainapp.parameters
        # mainapp = scafan.Scaffan()
        dct = scaffan.dilipg.params_and_values(p)
        self.assertEqual(type(dct), dict)

        self.assertIn("Output;Directory Path", dct)


def test_get_and_set_params():
    param_key = "Output;Directory Path"
    mainapp = scaffan.algorithm.Scaffan()
    param_val = mainapp.get_parameter(param_key)
    ini_str = mainapp.get_parameters_as_cfg_string()
    assert ini_str.find(param_key) > 0

    # set some parameters
    ini_str = "" +\
"""
[scaffan]
Output;Directory Path = /dir/with/nothing
"""
    mainapp.load_parameters_from_cfg_string(ini_str)
    param_val2 = mainapp.get_parameter(param_key)
    assert param_val != param_val2
    assert param_val2 == "/dir/with/nothing"

    # write again the same parameters
    mainapp.load_parameters_from_cfg_string(ini_str)
    param_val2 = mainapp.get_parameter(param_key)
    assert param_val == param_val


def test_load_parameters_from_cfg_file():
    param_key = "Output;Directory Path"
    mainapp = scaffan.algorithm.Scaffan()
    fn0 = "test_params0.cfg"
    fn1 = "test_params1.cfg"

    param_val0 = mainapp.get_parameter(param_key)
    mainapp.save_parameters_as_cfg_file(fn0)

    # set some parameters
    ini_str = """[scaffan]
Output;Directory Path = /dir/with/nothing
"""
    with open(fn1, 'w') as f:
        f.write(ini_str)

    mainapp.load_parameters_from_cfg_file(fn1)
    assert mainapp.get_parameter(param_key) == "/dir/with/nothing"

    mainapp.load_parameters_from_cfg_file(fn0)
    assert mainapp.get_parameter(param_key) == param_val0

