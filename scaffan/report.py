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
from pathlib import Path


class Report:
    def __init__(self, outputdir):
        # self.outputdir = op.expanduser(outputdir)
        self.outputdir = Path(outputdir).expanduser()
        if not op.exists(self.outputdir):
            os.makedirs(self.outputdir)

        self.df = pd.DataFrame()
        self.imgs = {}
        self.actual_row = {}
        self.show = False
        self.save = False

    def set_show(self, show):
        self.show = show

    def set_save(self, save):
        self.save = save

    def add_cols_to_actual_row(self, data):
        self.actual_row.update(data)

    # def write_table(self, filename):
    def finish_actual_row(self):
        data = self.actual_row
        df = pd.DataFrame([list(data.values())], columns=list(data.keys()))
        self.df = self.df.append(df, ignore_index=True)
        self.actual_row = {}

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

        if self.save:
            plt.imsave(op.join(self.outputdir, base_fn), arr)
        filename, ext = op.splitext(base_fn)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*low contrast image.*")
            # warnings.simplefilter("low contrast image")
            if self.save:
                skimage.io.imsave(op.join(self.outputdir, filename + "_raw" + ext), k * arr)
        self.imgs[base_fn] = arr

    def imsave_as_fig(self, base_fn, arr):
        filename, ext = op.splitext(base_fn)
        fig = plt.figure()
        plt.imshow(arr)
        plt.colorbar()
        if self.save:
            plt.savefig(op.join(self.outputdir, filename + "" + ext))
        if self.show:
            plt.show()
        else:
            plt.close(fig)

    # def add_array(self, base_fn, arr, k=50):
    #     if self.save:
    #         self.imsave

    def add_fig(self, base_fn):
        filename, ext = op.splitext(base_fn)
        if self.save:
            plt.savefig(op.join(self.outputdir, filename + "" + ext))
        if self.show:
            plt.show()
