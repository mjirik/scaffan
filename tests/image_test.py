#! /usr/bin/python
# -*- coding: utf-8 -*-

from loguru import logger
import unittest
import os.path as op
import pytest

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


class ImageAnnotationTest(unittest.TestCase):
    def test_get_pixelsize_on_different_levels(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        logger.debug("filename {}".format(fn))
        imsl = openslide.OpenSlide(fn)

        pixelsize1, pixelunit1 = scim.get_pixelsize(imsl)
        self.assertGreater(pixelsize1[0], 0)
        self.assertGreater(pixelsize1[1], 0)

        pixelsize2, pixelunit2 = scim.get_pixelsize(imsl, level=2)
        self.assertGreater(pixelsize2[0], 0)
        self.assertGreater(pixelsize2[1], 0)

        self.assertGreater(pixelsize2[0], pixelsize1[0])
        self.assertGreater(pixelsize2[1], pixelsize1[1])

    def test_anim(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        offset = anim.get_offset_px()
        self.assertEqual(len(offset), 2, "should be 2D")
        im = anim.get_image_by_center((10000, 10000), as_gray=True)
        self.assertEqual(len(im.shape), 2, "should be 2D")

        annotations = anim.read_annotations()
        self.assertGreater(len(annotations), 1, "there should be 2 annotations")
        assert im[0, 0] == pytest.approx(
            0.767964705882353, 0.001
        )  # expected intensity is 0.76
        # assert np.abs(im[0, 0] - 0.767964705882353) < 0.001  # expected intensity is 0.76

    def test_file_info(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        msg = anim.get_file_info()
        self.assertEqual(type(msg), str)
        self.assertLess(0, msg.find("mm"))

    def test_region(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        anim.set_region_on_annotations(0, 3)
        mask = anim.get_annotation_region_raster(0)
        image = anim.get_region_image()
        plt.imshow(image)
        plt.contour(mask)
        # plt.show()
        self.assertGreater(np.sum(mask), 20)
        assert image[0, 0, 0] == pytest.approx(97, 10)
        assert image[0, 0, 1] == pytest.approx(77, 10)
        assert image[0, 0, 2] == pytest.approx(114, 10)

        # this works on windows
        # assert image[0, 0, 0] == 97
        # assert image[0, 0, 1] == 77
        # assert image[0, 0, 2] == 114

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
        self.assertTrue(
            np.array_equal(mask.shape[:2], image.shape[:2]),
            "shape of mask should be the same as shape of image",
        )
        assert image[0, 0, 0] == pytest.approx(187, 10)

    # def test_region_select_area_definition_in_mm(self):
    #     fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
    #     anim = scim.AnnotatedImage(fn)
    #     anim.set_region_on_annotations("obj1", 3)
    #     mask = anim.get_annotation_region_raster("obj1")
    #     image = anim.get_region_image()
    #     plt.imshow(image)
    #     plt.contour(mask)
    #     # plt.show()
    #     self.assertGreater(np.sum(mask), 20)
    #     self.assertTrue(np.array_equal(mask.shape[:2], image.shape[:2]), "shape of mask should be the same as shape of image")

    def test_select_view_by_title_and_plot(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        annotation_ids = anim.select_annotations_by_title("obj1")
        view = anim.get_views(annotation_ids)[0]
        image = view.get_region_image()
        plt.imshow(image)
        view.plot_annotations("obj1")
        # plt.show()
        self.assertGreater(image.shape[0], 100)
        mask = view.get_annotation_raster("obj1")
        self.assertTrue(
            np.array_equal(mask.shape[:2], image.shape[:2]),
            "shape of mask should be the same as shape of image",
        )
        assert image[0, 0, 0] == 202

    def test_select_view_by_title_and_plot_floating_resolution(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        annotation_ids = anim.select_annotations_by_title("obj1")
        view = anim.get_views(annotation_ids)[0]
        pxsize, pxunit = view.get_pixelsize_on_level()
        image = view.get_region_image()
        plt.subplot(221)
        plt.imshow(image)
        view.plot_annotations("obj1")
        plt.suptitle("{} x {} [{}]".format(pxsize[0], pxsize[1], pxunit))

        self.assertGreater(image.shape[0], 100)
        mask = view.get_annotation_raster("obj1")
        self.assertTrue(
            np.array_equal(mask.shape[:2], image.shape[:2]),
            "shape of mask should be the same as shape of image",
        )
        plt.subplot(222)
        plt.imshow(mask)
        assert np.sum(mask == 1) == 874739  # number of expected pixels in mask

        view2 = view.to_pixelsize(pixelsize_mm=[0.01, 0.01])
        image2 = view2.get_region_image()
        plt.subplot(223)
        plt.imshow(image2)
        view2.plot_annotations("obj1")
        mask = view2.get_annotation_raster("obj1")
        plt.subplot(224)
        plt.imshow(mask)
        self.assertTrue(
            np.array_equal(mask.shape[:2], image2.shape[:2]),
            "shape of mask should be the same as shape of image",
        )
        assert np.sum(mask == 1) == 1812

        # plt.show()

    def test_merge_views(self):
        """
        Create two views with based on same annotation with different margin. Resize the inner to low resolution.
        Insert the inner image with low resolution into big image on high resolution.
        :return:
        """
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        annotation_ids = anim.select_annotations_by_title("obj1")
        view1 = anim.get_views(annotation_ids, margin=1.0, pixelsize_mm=[0.005, 0.005])[
            0
        ]
        image1 = view1.get_region_image()
        # plt.imshow(image1)
        # plt.colorbar()
        # plt.show()

        view2 = anim.get_views(annotation_ids, margin=0.1, pixelsize_mm=[0.05, 0.05])[0]
        image2 = view2.get_region_image()
        # plt.imshow(image2)
        # plt.show()
        logger.debug(
            f"Annotation ID: {annotation_ids}, location view1 {view1.region_location}, view2 {view2.region_location}"
        )

        merged = view1.insert_image_from_view(view2, image1, image2)
        # plt.imshow(merged)
        # plt.show()
        diffim = image1[:, :, :3].astype(np.int16) - merged[:, :, :3].astype(np.int16)
        errimg = np.mean(np.abs(diffim), 2)

        err = np.mean(errimg)
        self.assertLess(
            err, 3, "Mean error in intensity levels per pixel should be low"
        )
        self.assertLess(
            1,
            err,
            "Mean error in intensity levels per pixel should be low but there should be some error.",
        )

    def test_view_margin_size(self):
        """
        Compare two same resolution images with different margin
        :return:
        """
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        annotation_ids = anim.select_annotations_by_title("obj1")

        img1 = anim.get_views(annotation_ids, margin=0.0, pixelsize_mm=[0.005, 0.005])[
            0
        ].get_region_image(as_gray=True)
        img2 = anim.get_views(annotation_ids, margin=1.0, pixelsize_mm=[0.005, 0.005])[
            0
        ].get_region_image(as_gray=True)

        sh1 = np.asarray(img1.shape)
        sh2 = np.asarray(img2.shape)
        self.assertTrue(
            np.all((sh1 * 2.9) < sh2),
            "Boundary adds 2*margin*size of image to the image size",
        )
        self.assertTrue(
            np.all(sh2 < (sh1 * 3.1)),
            "Boundary adds 2*margin*size of image to the image size",
        )

        # test that everything is still pixel-precise
        assert img1[0, 0] == pytest.approx(
            0.47553590, 0.001
        )  # expected intensity is 0.76
        assert img2[0, 0] == pytest.approx(0.765724, 0.001)

    def test_select_view_by_center_mm(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        view = anim.get_view(
            center_mm=[10, 11],
            size_on_level=[100, 100],
            # level=5,
            # size_mm=[0.1, 0.1]
        )
        image = view.get_region_image()
        logger.debug(f"location: {view.region_location}")
        logger.debug(f"pixelsize: {view.region_pixelsize}")
        plt.imshow(image)
        # view.plot_annotations("obj1")
        # plt.show()

        assert image.shape[0] > 20
        assert image.shape[1] > 20

    def test_select_view_by_loc_mm(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        view = anim.get_view(
            location_mm=[10, 11],
            size_on_level=[100, 100],
            # level=5,
            # size_mm=[0.1, 0.1]
        )
        image = view.get_region_image()
        logger.debug(f"location: {view.region_location}")
        logger.debug(f"pixelsize: {view.region_pixelsize}")
        plt.imshow(image)
        # view.plot_annotations("obj1")
        # plt.show()

        assert image.shape[0] > 20
        assert image.shape[1] > 20
        # self.assertGreater(image.shape[0], 100)
        # mask = view.get_annotation_region_raster("obj1")
        # self.assertTrue(np.array_equal(mask.shape[:2], image.shape[:2]),
        #                 "shape of mask should be the same as shape of image")
        # assert image[0, 0, 0] == 202

    def test_outer_and_inner_annotation(self):
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        anim = scim.AnnotatedImage(fn)
        ann_ids = anim.select_annotations_by_color("#00FFFF")
        assert len(ann_ids) > 0
        assert len(ann_ids) == 2

        # find outer annotation from 0th cyan object
        outer_id = anim.select_outer_annotations(ann_ids[0])
        assert len(outer_id) == 1

        # find inner annotations to outer annotation of 0th object
        inner_ids = anim.select_inner_annotations(outer_id[0])
        assert len(inner_ids) == 2

        # find black inner annotations to outer annotation of 0th object
        inner_ids = anim.select_inner_annotations(outer_id[0], color="#000000")
        assert len(inner_ids) == 1

        cyan_inner_ids = anim.select_inner_annotations(outer_id[0], color="#00FFFF")
        assert len(cyan_inner_ids) == 1
        assert ann_ids[0] == cyan_inner_ids[0]

        black_ann_ids = anim.select_annotations_by_color("#000000")
        # find black inner annotations to outer annotation of 0th object
        inner_ids = anim.select_inner_annotations(outer_id[0], ann_ids=black_ann_ids)
        assert len(inner_ids) == 1

    def test_get_outer_ann(self):
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        anim = scim.AnnotatedImage(fn)
        color = "#000000"
        outer_ids, holes_ids = anim.select_just_outer_annotations(
            color, return_holes=True
        )
        logger.debug(f"outer ids {outer_ids}")
        logger.debug(f"holes ids {holes_ids}")
        assert len(outer_ids) > 0
        assert len(holes_ids) > 0
        assert len(holes_ids[0]) > 0
        assert (
            anim.select_inner_annotations(outer_ids[0], color=color)[0]
            == holes_ids[0][0]
        )

    def test_get_annotation_center(self):
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        anim = scim.AnnotatedImage(fn)
        center_x, center_y = anim.get_annotation_center_mm(1)
        # print(anim.annotations[1]["y_mm"])
        # print(anim.annotations[1]["x_mm"])
        # for some strange reason the y is usually negative
        assert center_x > -100
        assert center_y > -100
        assert center_x < 100
        assert center_y < 100

    def test_get_ann_by_color(self):
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        anim = scim.AnnotatedImage(fn)
        ann_ids_black = anim.select_annotations_by_color("#000000")
        assert 10 in ann_ids_black
        assert 11 in ann_ids_black

    def test_just_outer_annotations(self):
        fn = io3d.datasets.join_path(
            "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
        )
        anim = scim.AnnotatedImage(fn)
        outer_ids, holes_ids = anim.select_just_outer_annotations("#000000")
        assert 10 in outer_ids
        assert [11] in holes_ids

        views = anim.get_views(outer_ids, level=6)
        # ann_rasters = []
        for id1, id2, view_ann in zip(outer_ids, holes_ids, views):
            ann_raster = view_ann.get_annotation_raster(id1, holes_ids=id2)
            # ann_rasters.append(ann_raster)
            assert np.min(ann_raster) == 0
            assert np.max(ann_raster) > 0
