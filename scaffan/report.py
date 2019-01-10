# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process lobulus analysis.
"""
import logging

logger = logging.getLogger(__name__)
import pandas as pd
import os.path as op
import os
import matplotlib.pyplot as plt
import skimage.io
import warnings


class Report:
    def __init__(self, outputdir):
        self.outputdir = op.expanduser(outputdir)
        if not op.exists(self.outputdir):
            os.makedirs(self.outputdir)

        self.df = pd.DataFrame()
        self.imgs = {}

    def add_row(self, data):
        df = pd.DataFrame([list(data.values())], columns=list(data.keys()))
        self.df = self.df.append(df, ignore_index=True)

    # def write_table(self, filename):

    def add_table(self):
        pass

    def write(self):
        self.df.to_excel(op.join(self.outputdir, "data.xlsx"))

    def imsave(self, base_fn, arr, k=50):
        """
        :param base_fn: with a format slot for annotation id like "skeleton_{}.png"
        :param arr:
        :return:
        """
        plt.imsave(op.join(self.outputdir, base_fn), arr)
        filename, ext = op.splitext(base_fn)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*low contrast image.*")
            # warnings.simplefilter("low contrast image")
            skimage.io.imsave(op.join(self.outputdir, filename + "_raw" + ext), k * arr)
        self.imgs[base_fn] = arr
