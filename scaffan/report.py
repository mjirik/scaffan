# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process lobulus analysis.
"""
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import os.path as op
import sys
import os


class Report():
    def __init__(self, outputdir):
        self.outputdir = op.expanduser(outputdir)
        if not op.exists(self.outputdir):
            os.makedirs(self.outputdir)



    def add_table(self):
        pass
