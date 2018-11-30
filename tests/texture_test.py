#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
import unittest
import os.path as op
from nose.plugins.attrib import attr

path_to_script = op.dirname(op.abspath(__file__))

import sys

sys.path.insert(0, op.abspath(op.join(path_to_script, "../../io3d")))
sys.path.insert(0, op.abspath(op.join(path_to_script, "../../imma")))
# import sys
# import os.path

# imcut_path =  os.path.join(path_to_script, "../../imcut/")
# sys.path.insert(0, imcut_path)

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

skip_on_local = False

import scaffan.image as scim
scim.import_openslide()
import openslide

import io3d
import scaffan
import scaffan.image as scim
import scaffan.texture as satex
import scaffan.texture_lbp as salbp
from scaffan.texture_lbp import local_binary_pattern


class TextureTest(unittest.TestCase):


    def test_region_select_by_title(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        anim.set_region_on_annotations("obj1", 3)
        mask = anim.get_annotation_region_raster("obj1")
        image = anim.get_region_image()
        plt.imshow(image)
        plt.contour(mask)
        # plt.show()
        self.assertGreater(np.sum(mask), 20)

    def test_select_by_title_and_plot(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        anim.set_region_on_annotations("obj1", level=3)
        image = anim.get_region_image()
        plt.imshow(image)
        anim.plot_annotations("obj1")
        # plt.show()
        self.assertGreater(image.shape[0], 100)

    def test_select_view_by_title_and_plot(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        view = anim.get_view_on_annotation("obj1", 3)
        image = view.get_region_image()
        plt.imshow(image)
        view.plot_annotations("obj1")
        # plt.show()
        self.assertGreater(image.shape[0], 100)
        mask = view.get_annotation_region_raster("obj1")
        self.assertTrue(np.array_equal(mask.shape[:2], image.shape[:2]), "shape of mask should be the same as shape of image")

        nz = satex.select_texture_patch_centers_from_one_annotation(anim, "obj1", tile_size=32, level=3, step=20)
        nz_view_px = view.coords_glob_px_to_view_px(nz)
        plt.plot(nz_view_px[1], nz_view_px[0], "bo")
        # plt.show()

        x = nz_view_px[0].astype(int)
        y = nz_view_px[1].astype(int)
        pixels = mask[(x, y)]
        self.assertTrue(np.all(pixels > 0), "centers positions should be inside of mask")

    def test_simple_texture_segmentation(self):
        level=0
        title_size = 128
        size = [128, 128]
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        patch_centers0 = satex.select_texture_patch_centers_from_one_annotation(anim, "obj_empty", tile_size=title_size, level=level, step=64)
        patch_centers1 = satex.select_texture_patch_centers_from_one_annotation(anim, "obj1", tile_size=title_size, level=level, step=64)
        patch_centers2 = satex.select_texture_patch_centers_from_one_annotation(anim, "obj2", tile_size=title_size, level=level, step=64)
        patch_centers3 = satex.select_texture_patch_centers_from_one_annotation(anim, "obj3", tile_size=title_size, level=level, step=64)
        view0 = anim.get_view(center=[patch_centers0[0][0], patch_centers0[1][0]], level=level, size=size)
        view1 = anim.get_view(center=[patch_centers1[0][0], patch_centers1[1][0]], level=level, size=size)
        view2 = anim.get_view(center=[patch_centers2[0][0], patch_centers2[1][0]], level=level, size=size)
        view3 = anim.get_view(center=[patch_centers3[0][0], patch_centers3[1][0]], level=level, size=size)

        plt.imshow(view1.get_region_image())
        plt.figure()
        plt.imshow(view2.get_region_image())
        plt.figure()
        plt.imshow(view3.get_region_image())
        # plt.show()
        im0 = view0.get_region_image(as_gray=True)
        im1 = view1.get_region_image(as_gray=True)
        im2 = view2.get_region_image(as_gray=True)
        im3 = view3.get_region_image(as_gray=True)
        radius = 3
        n_points = 8
        METHOD = "uniform"
        refs = {
            0: local_binary_pattern(im0, n_points, radius, METHOD),
            1: local_binary_pattern(im1, n_points, radius, METHOD),
            2: local_binary_pattern(im2, n_points, radius, METHOD),
            3: local_binary_pattern(im3, n_points, radius, METHOD)
        }
        refs = [
            [0, local_binary_pattern(im0, n_points, radius, METHOD)],
            [1, local_binary_pattern(im1, n_points, radius, METHOD)],
            [2, local_binary_pattern(im2, n_points, radius, METHOD)],
            [3, local_binary_pattern(im3, n_points, radius, METHOD)]
        ]
        view_test = anim.get_view_on_annotation("test2", level=level)
        test_image = view_test.get_region_image(as_gray=True)

        seg = satex.texture_segmentation(test_image, salbp.match, refs, tile_size=size)
        plt.figure()
        plt.imshow(test_image)
        plt.contour(seg)
        import skimage.color
        plt.figure()
        plt.imshow(skimage.color.label2rgb(seg, test_image))
        # plt.show()



    def test_texture_segmentation_object(self):
        level=0
        title_size = 128
        size = [128, 128]
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)

        texseg = satex.TextureSegmentation()
        texseg.add_training_data(anim, "obj1", 1)
        texseg.add_training_data(anim, "obj2", 2)
        texseg.add_training_data(anim, "obj3", 3)

        texseg.fit(anim.get_view_on_annotation("test2", level=1), show=True )
        # plt.show()


if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
