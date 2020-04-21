#! /usr/bin/python
# -*- coding: utf-8 -*-

from loguru import logger
import unittest
import os.path as op
from pathlib import Path
import pytest

# path_to_script = op.dirname(op.abspath(__file__))
path_to_script = Path(__file__).absolute().parent

import sys

sys.path.insert(0, op.abspath(op.join(path_to_script, "../../io3d")))
sys.path.insert(0, op.abspath(op.join(path_to_script, "../../imma")))
# import sys
# import os.path

# imcut_path =  os.path.join(path_to_script, "../../imcut/")
# sys.path.insert(0, imcut_path)

import numpy as np
import matplotlib.pyplot as plt

skip_on_local = False

import scaffan.image as scim
from scaffan import image_czi

scim.import_openslide()
import io3d
from czifile import CziFile


def test_read_czi():

    fn = io3d.datasets.join_path("medical/orig/scaffan-analysis-czi/J7_5/J7_5_b.czi", get_root=True)
    # fn = io3d.datasets.join_path(
    #     "medical/orig/scaffan-analysis-czi/Zeiss-scans/01_2019_11_12__RecognizedCode.czi",
    #     get_root=True)
    logger.debug("filename {}".format(fn))
    anim = scim.AnnotatedImage(fn)
    size_px = 100
    size_on_level = [size_px, size_px]
    view = anim.get_view(
        location_mm=[10, 11],
        size_on_level=size_on_level,
        level=5,
        # size_mm=[0.1, 0.1]
    )
    img = view.get_region_image(as_gray=True)
    # plt.imshow(img)
    # plt.show()
    assert img.shape[0] == size_px
    assert img.shape[1] == size_px
    # offset = anim.get_offset_px()
    # assert len(offset) == 2, "should be 2D"
    im = anim.get_image_by_center((5000, 5000), as_gray=True)
    assert len(im.shape) == 2, "should be 2D"

    annotations = anim.read_annotations()
    assert len(annotations) == 0, "there should be 0 annotations"
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # assert im[0, 0] == pytest.approx(
    #     0.767964705882353, 0.001
    # )  # expected intensity is 0.76
    # imsl = openslide.OpenSlide(fn)
    imsl = anim.openslide

    pixelsize1, pixelunit1 = scim.get_pixelsize(imsl)
    assert pixelsize1[0] > 0
    assert pixelsize1[1] > 0

    pixelsize2, pixelunit2 = scim.get_pixelsize(imsl, level=2)
    assert pixelsize2[0] > 0
    assert pixelsize2[1] > 0

    assert pixelsize2[0] > pixelsize1[0]
    assert pixelsize2[1] > pixelsize1[1]


def test_read_czi_per_partes():

    fn = io3d.datasets.join_path(
        "medical/orig/scaffan-analysis-czi/Zeiss-scans/01_2019_11_12__RecognizedCode.czi",
        get_root=True)
    requested_start = [23000, -102000]
    requested_size = [1000, 1000]
    # requested_size = [20, 20]
    requested_level = 0

    with CziFile(fn) as czi:
        output1 = image_czi.read_region_with_scale(czi, requested_start, size=[400, 500], downscale_factor=2)
        output0 = image_czi.read_region_with_scale(czi, requested_start, size=[800, 1000], downscale_factor=1)

    bins = list(range(0, 260, 20))
    dens0,_ = np.histogram(output0[:], bins=bins, density=True)
    dens1,_ = np.histogram(output1[:], bins=bins, density=True)
    assert pytest.approx(dens0, abs=0.1) == dens1, "Relative histograms should be almost equal"
    # plt.figure()
    # plt.imshow(output0.astype(np.uint8))
    # plt.figure()
    # plt.imshow(output1.astype(np.uint8))
    # plt.show()

def test_read_czi_with_scaffold_data():

    # fn = io3d.datasets.join_path("medical/orig/scaffan-analysis-czi/J7_5/J7_5_b.czi", get_root=True)
    fn = io3d.datasets.join_path(
        "medical/orig/scaffan-analysis-czi/Zeiss-scans/01_2019_11_12__RecognizedCode.czi",
        get_root=True)
    logger.debug("filename {}".format(fn))
    anim = scim.AnnotatedImage(fn)
    size_px = 1000
    size_on_level = [size_px, size_px]
    view = anim.get_view(
        location_mm=[10, 51],
        size_on_level=size_on_level,
        level=4,
        # size_mm=[0.1, 0.1]
    )
    img = view.get_region_image(as_gray=False)
    # plt.imshow(img)
    # plt.show()
    assert img.shape[0] == size_px
    assert img.shape[1] == size_px
    # offset = anim.get_offset_px()
    # assert len(offset) == 2, "should be 2D"
    im = anim.get_image_by_center((1500, 76600), as_gray=True)
    assert len(im.shape) == 2, "should be 2D"

    annotations = anim.read_annotations()
    assert len(annotations) == 0, "there should be 0 annotations"
    # assert im[0, 0] == pytest.approx(
    #     0.767964705882353, 0.001
    # )  # expected intensity is 0.76
    # imsl = openslide.OpenSlide(fn)
    imsl = anim.openslide

    pixelsize1, pixelunit1 = scim.get_pixelsize(imsl)
    assert pixelsize1[0] > 0
    assert pixelsize1[1] > 0

    logger.debug(f"pixelsize={pixelsize1}")
    # plt.figure()
    # plt.imshow(im, cmap='gray')
    # plt.show()

    pixelsize2, pixelunit2 = scim.get_pixelsize(imsl, level=2)
    assert pixelsize2[0] > 0
    assert pixelsize2[1] > 0

    assert pixelsize2[0] > pixelsize1[0]
    assert pixelsize2[1] > pixelsize1[1]
