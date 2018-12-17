# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging
logger = logging.getLogger(__name__)
# problem is loading lxml together with openslide
# from lxml import etree
from typing import List
import json
import os.path as op
import glob
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import skimage.color
from . import annotation as scan
from matplotlib.path import Path


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

    if resolution_unit in ("cm", "centimeter"):
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
    # resolution_unit = pm["tiff.ResolutionUnit"]
    offset_mm = offset * 0.000001
    if pixelunit is not "mm":
        raise ValueError("Cannot convert pixelunit {} to milimeters".format(pixelunit))
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
        self.titles = scan.annotation_titles(self.annotations)
        self.colors = scan.annotation_colors(self.annotations)
        return self.annotations

    def get_view(self, center=None, level=0, size=None, location=None):
        view = View(anim=self, center=center, level=level, size=size, location=location)
        return view

    def get_views_by_title(self, title=None, level=2, return_ids=False, **kwargs):
        annotation_ids = self.get_annotation_ids(title)
        if return_ids:
            return self.get_views(annotation_ids, level=level, **kwargs), annotation_ids
        else:
            return self.get_views(annotation_ids, level=level, **kwargs)

    def select_annotations_by_title(self, title):
        return self.get_annotation_ids(title)
        # return self.get_views(annotation_ids, level=level, **kwargs), annotation_ids

    def get_views_by_annotation_color(self):
        pass

    def get_views(self, annotation_ids=None, level=2, margin=0.5, margin_in_pixels=False, show=False) -> List['View']:
        views = [None] * len(annotation_ids)
        for i, annotation_id in enumerate(annotation_ids):
            center, size = self.get_annotations_bounds_px(annotation_id)
            if margin_in_pixels:
                margin_px = int(margin)
            else:
                margin_px = (size * margin).astype(np.int)
            region_size = ((size / self.openslide.level_downsamples[level]) + 2 * margin_px).astype(int)
            view = self.get_view(center=center, level=level, size=region_size)
            if show:
                view.region_imshow_annotation(annotation_id)
            views[i] = view

        return views

    def set_region(self, center=None, level=0, size=None, location=None):

        if size is None:
            size = [256, 256]

        size = np.asarray(size)

        if location is None:
            location = self.get_region_location_by_center(center, level, size)
        else:
            center = self.get_region_center_by_location(location, level, size)

        self.region_location = location
        self.region_center = center
        self.region_size = size
        self.region_level = level
        self.region_pixelsize, self.region_pixelunit = self.get_pixel_size(level)
        scan.adjust_annotation_to_image_view(self.openslide, self.annotations,
                                             center, level, size)

    def select_annotations_by_color(self, id):
        if type(id) is str:
            id = self.colors[id]
        else:
            id = [id]
        return id

    def get_annotation_ids(self, id):
        if type(id) is str:
            id = self.titles[id]
        else:
            id = [id]
        return id

    def get_annotation_id(self, i):
        if type(i) is str:
            i = self.titles[i][0]
        return i

    def set_region_on_annotations(self, i=None, level=2, boundary_px=10, show=False):
        """

        :param i: index of annotation or annotation title
        :param level:
        :param boundary_px:
        :return:
        """
        i = self.get_annotation_id(i)
        center, size = self.get_annotations_bounds_px(i)
        region_size = ((size / self.openslide.level_downsamples[level]) + 2 * boundary_px).astype(int)
        self.set_region(center=center, level=level, size=region_size)
        if show:
            self.region_imshow_annotation(i)

    def get_annotations_bounds_px(self, i=None):
        i = self.get_annotation_id(i)
        if i is not None:
            anns = [self.annotations[i]]

        x_px = []
        y_px = []

        for ann in anns:
            x_px.append(ann["x_px"])
            y_px.append(ann["y_px"])

        mx = np.array([np.max(x_px), np.max(y_px)])
        mi = np.array([np.min(x_px), np.min(y_px)])
        all = [mi, mx]
        center = np.mean(all, 0)
        size = mx - mi
        return center, size

    def get_region_image(self, as_gray=False):
        imcr = self.openslide.read_region(
            self.region_location, level=self.region_level, size=self.region_size)
        im = np.asarray(imcr)
        if as_gray:
            im = skimage.color.rgb2gray(im)
        return im

    def plot_annotations(self, annotation_id=None):
        if annotation_id is None:
            anns = self.annotations
        else:
            annotation_id = self.get_annotation_id(annotation_id)
            anns = [self.annotations[annotation_id]]
        scan.plot_annotations(anns, in_region=True)

    def get_annotation_region_raster(self, i):
        i = self.get_annotation_id(i)
        polygon_x = self.annotations[i]["region_x_px"]
        polygon_y = self.annotations[i]["region_y_px"]
        polygon = list(zip(polygon_y, polygon_x))
        poly_path = Path(polygon)

        x, y = np.mgrid[:self.region_size[1], :self.region_size[0]]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))  # coors.shape is (4000000,2)

        mask = poly_path.contains_points(coors)
        mask = mask.reshape(self.region_size[::-1])
        return mask

    def region_imshow_annotation(self, i):
        region = self.get_region_image()
        plt.imshow(region)
        self.plot_annotations(i)

    def coords_region_px_to_global_px(self, points_view_px):
        """
        :param points_view_px: [[x0, x1, ...], [y0, y1, ...]]
        :return:
        """

        px_factor = self.openslide.level_downsamples[self.region_level]
        print(px_factor)
        x_px = self.region_location[0] + points_view_px[0] * px_factor
        y_px = self.region_location[1] + points_view_px[1] * px_factor

        return x_px, y_px

    def coords_global_px_to_view_px(self, points_glob_px):
        """
        :param points_glob_px: [[x0, x1, ...], [y0, y1, ...]]
        :return:
        """

        px_factor = self.openslide.level_downsamples[self.region_level]
        print(px_factor)
        x_glob_px = points_glob_px[0]
        y_glob_px = points_glob_px[1]
        x_view_px = (x_glob_px - self.region_location[0]) / px_factor
        y_view_px = (y_glob_px - self.region_location[1]) / px_factor

        return x_view_px, y_view_px

class View:

    def __init__(self, anim, center=None, level=0, size=None, location=None):
        self.anim = anim
        self.set_region(center=center, level=level, size=size, location=location)

    def set_region(self, center=None, level=0, size=None, location=None):

        if size is None:
            size = [256, 256]

        size = np.asarray(size)

        if location is None:
            location = self.get_region_location_by_center(center, level, size)
        else:
            center = self.get_region_center_by_location(location, level, size)

        self.region_location = location
        self.region_center = center
        self.region_size = size
        self.region_level = level
        self.region_pixelsize, self.region_pixelunit = self.get_pixel_size(level)
        import copy
        self.annotations = copy.deepcopy(self.anim.annotations)
        scan.adjust_annotation_to_image_view(self.anim.openslide, self.annotations,
                                             center, level, size)

    def get_pixel_size(self, level=None):
        if level is None:
            level = self.region_level
        return self.anim.get_pixel_size(level)

    def get_region_image(self, as_gray=False):
        imcr = self.openslide.read_region(
            self.region_location, level=self.region_level, size=self.region_size)
        im = np.asarray(imcr)
        if as_gray:
            im = skimage.color.rgb2gray(im)
        return im

    def region_imshow_annotation(self, i):
        region = self.get_region_image()
        plt.imshow(region)
        self.plot_annotations(i)

    def coords_glob_px_to_view_px(self, x_glob_px, y_glob_px):
        px_factor = self.anim.openslide.level_downsamples[self.region_level]

        x_px = (x_glob_px - self.region_location[0]) / px_factor
        y_px = (y_glob_px - self.region_location[1]) / px_factor

        return x_px, y_px

    def coords_view_px_to_glob_px(self, x_view_px, y_view_px):
        """
        :param x_view_px: [x0, x1, ...]
        :param y_view_px: [y0, y1, ...]]
        :return:
        """
        px_factor = self.anim.openslide.level_downsamples[self.region_level]
        # print(px_factor)
        x_px = self.region_location[0] + x_view_px * px_factor
        y_px = self.region_location[1] + y_view_px * px_factor

        return x_px, y_px

    def plot_points(self, x_glob_px, y_glob_px):
        # points = [x_glob_px, y_glob_px]
        x_view_px, y_view_px = self.coords_glob_px_to_view_px(x_glob_px, y_glob_px)
        plt.plot(x_view_px, y_view_px, "oy")

    def get_annotation_region_raster(self, annotation_id):
        annotation_id = self.anim.get_annotation_id(annotation_id)
        polygon_x = self.annotations[annotation_id]["region_x_px"]
        polygon_y = self.annotations[annotation_id]["region_y_px"]
        polygon = list(zip(polygon_y, polygon_x))
        poly_path = Path(polygon)

        # TODO coordinates are swapped here
        x, y = np.mgrid[:self.region_size[1], :self.region_size[0]]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))  # coors.shape is (4000000,2)

        mask = poly_path.contains_points(coors)
        mask = mask.reshape(self.region_size[::-1])
        return mask

    def region_imshow_annotation(self, i):
        region = self.get_region_image()
        plt.imshow(region)
        self.plot_annotations(i)

    def plot_annotations(self, i=None):
        if i is None:
            anns = self.annotations
        else:
            i = self.anim.get_annotation_id(i)
            anns = [self.annotations[i]]
        scan.plot_annotations(anns, in_region=True)

    def get_region_location_by_center(self, center, level, size):
        return get_region_location_by_center(self.anim.openslide, center, level, size)

    def get_region_image(self, as_gray=False):
        imcr = self.anim.openslide.read_region(
            self.region_location, level=self.region_level, size=self.region_size)
        im = np.asarray(imcr)
        if as_gray:
            im = skimage.color.rgb2gray(im)
        return im

    def imshow(self, as_gray=False):
        plt.imshow(self.get_region_image(as_gray=as_gray))

