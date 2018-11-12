# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging
logger = logging.getLogger(__name__)
# problem is loading lxml together with openslide
# from lxml import etree
import json
import os.path as op
import glob
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import skimage.color

def import_openslide():
    pth = op.expanduser(r"~\Downloads\openslide\openslide-win64\bin")
    # pth = op.expanduser(r"~\projects\scaffan\devel\knihovny")
    # pth = op.expanduser(r"~\Miniconda3\envs\lisa36\Library\bin")
    sys.path.insert(0, pth)
    orig_PATH = os.environ["PATH"]
    os.environ["PATH"] = pth + ";" + os.environ["PATH"]
    import openslide

# import_openslide()


# def
def get_image_with_center(imsl, center, level=3, size=None, as_gray=True):
    if size is None:
        size = np.array([800, 800])
    size2 = (size/2).astype(int)

    offset = size2 * imsl.level_downsamples[level]
    location = (np.asarray(center) - offset).astype(np.int)

    imcr = imsl.read_region(location, level=level, size=size)
    im = np.asarray(imcr)
    if as_gray:
        im = skimage.color.rgb2gray(im)
    return im
