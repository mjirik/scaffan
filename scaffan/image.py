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
import scaffan.annotation as scan


def import_openslide():
    pth = op.expanduser(r"~\Downloads\openslide\openslide-win64\bin")
    # pth = op.expanduser(r"~\projects\scaffan\devel\knihovny")
    # pth = op.expanduser(r"~\Miniconda3\envs\lisa36\Library\bin")
    sys.path.insert(0, pth)
    orig_PATH = os.environ["PATH"]
    orig_split = orig_PATH.split(";")
    if pth not in orig_split:
        print("add path {}".format(pth))
    os.environ["PATH"] = pth + ";" + os.environ["PATH"]
    import openslide

import_openslide()
import openslide


# def
def get_image_by_center(imsl, center, level=3, size=None, as_gray=True):
    if size is None:
        size = np.array([800, 800])

    location = get_region_location_by_center(imsl, center, level, size)

    imcr = imsl.read_region(location, level=level, size=size)
    im = np.asarray(imcr)
    if as_gray:
        im = skimage.color.rgb2gray(im)
    return im


def get_region_location_by_center(imsl, center, level, size):
    size2 = (size/2).astype(int)

    offset = size2 * imsl.level_downsamples[level]
    location = (np.asarray(center) - offset).astype(np.int)
    return location


def get_region_center_by_location(imsl, location, level, size):
    size2 = (size/2).astype(int)

    offset = size2 * imsl.level_downsamples[level]
    center = (np.asarray(location) + offset).astype(np.int)
    return center

def get_pixelsize(imsl, level=0):
    """
    imageslice
    :param imsl: image slice obtained by openslice.OpenSlide(path)
    :return: pixelsize, pixelunit
    """
    pm = imsl.properties
    resolution_unit = pm.get("tiff.ResolutionUnit")
    resolution_x= pm.get("tiff.XResolution")
    resolution_y= pm.get("tiff.YResolution")
#     print("Resolution {}x{} pixels/{}".format(resolution_x, resolution_y, resolution_unit))
    downsamples = imsl.level_downsamples[level]

    if resolution_unit is "cm":
        downsamples = downsamples * 10.
        pixelunit = "mm"
    else:
        pixelunit = resolution_unit

    pixelsize = [downsamples /float(resolution_x), downsamples /float(resolution_y)]

    return pixelsize, pixelunit


def get_offset_px(imsl):

    pm = imsl.properties
    pixelsize, pixelunit = get_pixelsize(imsl)
    offset = np.asarray((int(pm['hamamatsu.XOffsetFromSlideCentre']), int(pm['hamamatsu.YOffsetFromSlideCentre'])))
    offset_mm = offset * 0.000001
    offset_from_center_px = offset_mm / pixelsize
    im_center_px = np.asarray(imsl.dimensions) / 2.
    offset_px = im_center_px - offset_from_center_px
    return offset_px


def get_resize_parameters(imsl, former_level, former_size, new_level):
    """
    Get scale factor and size of image on different level.

    :param imsl: OpenSlide
    :param former_level: int
    :param former_size: list of ints
    :param new_level: int
    :return: scale_factor, new_size
    """
    scale_factor = imsl.level_downsamples[former_level] / imsl.level_downsamples[new_level]
    new_size = (np.asarray(former_size) * scale_factor).astype(np.int)
    return scale_factor, new_size


class AnnotatedImage:
    def __init__(self, path, skip_read_annotations=False):
        self.path = path
        self.openslide = openslide.OpenSlide(path)
        self.region_location = None
        self.region_size = None
        self.region_level = None
        self.region_pixelsize = None
        self.region_pixelunit = None

        if not skip_read_annotations:
            self.read_annotations()

    def get_resize_parameters(self, former_level, former_size, new_level):
        """
        Get scale and size of image after resize to other level
        :param former_level:
        :param former_size:
        :param new_level:
        :return: scale_factor, new_size
        """
        return get_resize_parameters(self.openslide, former_level, former_size, new_level)

    def get_offset_px(self):
        return get_offset_px(self.openslide)

    def get_pixel_size(self, level=0):
        return get_pixelsize(self.openslide, level)

    def get_image_by_center(self, center, level=3, size=None, as_gray=True):
        return get_image_by_center(self.openslide, center, level, size, as_gray)

    def get_region_location_by_center(self, center, level, size):
        return get_region_location_by_center(self.openslide, center, level, size)

    def get_region_center_by_location(self, location, level, size):
        return get_region_center_by_location(self.openslide, location, level, size)

    def read_annotations(self):
        self.annotations = scan.read_annotations(self.path)
        self.annotations = scan.annotations_to_px(self.openslide, self.annotations)
        return self.annotations

    def set_region(self, location=None, level=0, size=None, center=None):

        if location is None:
            location = self.get_region_location_by_center(center, level, size)
        else:
            center = self.get_region_center_by_location(location, level, size)

        self.region_location = location
        self.region_size = size
        self.region_level = level
        self.region_center = level
        self.region_pixelsize, self.region_pixelunit = self.get_pixel_size(level)
        scan.adjust_to_image_view(self.openslide, self.annotations,
                                  center, level, size)

    def get_region(self, as_gray=False):
        imcr = openslide.read_region(
            self.region_location, level=self.region_level, size=self.region_size)
        im = np.asarray(imcr)
        if as_gray:
            im = skimage.color.rgb2gray(im)
        return im


