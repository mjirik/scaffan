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

        self.df = pd.DataFrame()

    def add_row(self, data):
        df = pd.DataFrame([list(data.values())], columns=list(data.keys()))
        self.df = self.df.append(df, ignore_index=True)

    # def write_table(self, filename):


    def add_table(self):
        pass
