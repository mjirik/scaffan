#! /usr/bin/python
# -*- coding: utf-8 -*-

from loguru import logger
import unittest
import os
import io3d
from pathlib import Path
import shutil
import pandas as pd


# import openslide
import scaffan
import scaffan.algorithm
import pyqtgraph.parametertree

import scaffan.dilipg




class ParameterTest(unittest.TestCase):

    skip_on_local = False
    # skip_on_local = True
    # @unittest.skipIf(os.environ.get("TRAVIS", skip_on_local), "Skip on Travis-CI")
    def test_save_report_excel(self):
        mainapp = scaffan.algorithm.Scaffan()
        p = mainapp.parameters
        # mainapp = scaffan.Scaffan()
        dct = scaffan.dilipg.params_and_values(p)
        self.assertEqual(type(dct), dict)

        self.assertIn('Output;Directory Path', dct)

