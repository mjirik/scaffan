#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
import unittest
import os.path as op
# from nose.plugins.attrib import attr

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
        annotation_ids = anim.select_annotations_by_title("obj1")
        view = anim.get_views(annotation_ids, level=3)[0]
        image = view.get_region_image()
        plt.imshow(image)
        view.plot_annotations("obj1")
        # plt.show()
        self.assertGreater(image.shape[0], 100)
        mask = view.get_annotation_region_raster("obj1")
        self.assertTrue(np.array_equal(mask.shape[:2], image.shape[:2]), "shape of mask should be the same as shape of image")

        x_nz, y_nz = satex.select_texture_patch_centers_from_one_annotation(anim, "obj1", tile_size=32, level=3, step=20)
        nz_view_px = view.coords_glob_px_to_view_px(x_nz, y_nz)
        plt.plot(nz_view_px[0], nz_view_px[1], "bo")
        # plt.show()

        # TODO findout why are the axis swapped
        x = nz_view_px[1].astype(int)
        y = nz_view_px[0].astype(int)
        pixels = mask[(x, y)]
        self.assertTrue(np.all(pixels > 0), "centers positions should be inside of mask")

    def test_simple_texture_segmentation(self):
        level = 0
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
        annotation_ids = anim.select_annotations_by_title("test2")
        view_test = anim.get_views(annotation_ids, level=level)[0]
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

        level = 0
        tile_size1 = 128
        tile_size = [128, 128]
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)

        texseg = satex.TextureSegmentation()
        texseg.level = level
        texseg.tile_size = tile_size
        texseg.tile_size1 = tile_size1
        texseg.step = 128

        plt.figure()
        texseg.add_training_data(anim, "obj1", 1, show=True)
        texseg.show_tiles(anim, "obj1", [0, 1, -2, -1])
        plt.figure()
        texseg.add_training_data(anim, "obj2", 2, show=True)
        plt.figure()
        texseg.add_training_data(anim, "obj3", 3, show=True)
        # plt.show()
        views = anim.get_views_by_title("test2", level=texseg.level)
        texseg.fit(views[0], show=True)
        # plt.show()


    skip_on_local = True
    @unittest.skipIf(os.environ.get("TRAVIS", skip_on_local), "Skip on Travis-CI")
    def test_texture_segmentation_object_lobulus_data(self):
        fn = io3d.datasets.join_path("scaffold", "Hamamatsu", "PIG-008_P008 LL-P_HE_parenchyme perif..ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)

        texseg = satex.TextureSegmentation()
        # texseg.add_training_data(anim, "empty1", 0)
        plt.figure()
        texseg.add_training_data(anim, "intralobular1", 1, show=True)
        plt.figure()
        texseg.add_training_data(anim, "intralobular2", 1, show=True)
        # centers = texseg.get_tile_centers(anim, "intralobular1")
        # view1 = texseg.get_patch_view(anim, centers[0])
        # plt.imshow(view1.get_region_image())
        # view1 = texseg.get_patch_view(anim, centers[1])
        # plt.imshow(view1.get_region_image())
        # plt.show()
        plt.figure()
        # print("number of patches: {}".format(len(texseg.refs)))
        texseg.add_training_data(anim, "extralobular1", 2, show=True)
        plt.figure()
        texseg.add_training_data(anim, "extralobular3", 2, show=True)

        # print("number of patches: {}".format(len(texseg.refs)))

        texseg.show_tiles(anim, "intralobular1", [0, -1])
        # texseg.show_tiles(anim, "intralobular2", [0, -1])
        # texseg.show_tiles(anim, "extralobular1", [0, -1])
        texseg.show_tiles(anim, "extralobular3", [0, -1])
        # plt.show()
        logger.debug("number of patches: {}".format(len(texseg.refs)))
        # texseg.add_training_data(anim, "obj3", 3)

        views = anim.get_views_by_title("test3", level=texseg.level)
        texseg.fit(views[0], show=True)
        plt.savefig("segmentation.png")
        # plt.show()


if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
