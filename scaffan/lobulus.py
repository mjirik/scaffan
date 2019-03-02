# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process lobulus analysis.
"""
import logging

logger = logging.getLogger(__name__)
import skimage.filters
from skimage.morphology import skeletonize
import skimage.io
import scipy.signal
import scipy.ndimage
import os.path as op
import numpy as np
import warnings
import morphsnakes as ms
from matplotlib import pyplot as plt
from scaffan import image as scim
from .report import Report
from pyqtgraph.parametertree import Parameter
import imma.image

_cite = "" +\
    "[1]: A Morphological Approach to Curvature-based Evolution of Curves and Surfaces, Pablo Márquez-Neila, Luis Baumela and Luis Álvarez. In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2014, DOI 10.1109/TPAMI.2013.106" +\
    "[2]: Morphological Snakes. Luis Álvarez, Luis Baumela, Pablo Márquez-Neila. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2010 (CVPR10)."

class Lobulus:
    def __init__(self):
        # TODO the segmentation resolution was probably different.
        #  For segmentation was used different level than 2. Probably 3 or 4
        #  The level 2 has been used in detail view

        params = [
            # {
            #     "name": "Tile Size",
            #     "type": "int",
            #     "value" : 128
            # },
            {
                "name": "Working Resolution",
                "type": "float",
                # "value": 0.000001,
                # "value": 0.00000091, # this is typical resolution on level 2
                # "value": 0.00000182,  # this is typical resolution on level 3
                # "value": 0.00000364,  # this is typical resolution on level 4
                "value": 0.00000728,  # this is typical resolution on level 5
                # "value": 0.00001456,  # this is typical resolution on level 6
                "suffix": "m",
                "siPrefix": True

            },
            {
                "name": "Inner Lobulus Margin",
                "type": "float",
                "value": 0.00002,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Area close to the border is ignored in Otsu threshold computation before skeletonization step."
            },
            {
                'name': 'Border Segmentation', 'type': 'group',
                'tip': "MorphACWE algorithm parameters. " + _cite,
                'children': [
                    {
                        "name": "Smoothing",
                        "type": "float",
                        "value": 2,
                        "suffix": "px",
                        "siPrefix": False,
                        "tip": "MorphACWE algorithm parameter: The number of repetitions of the smoothing step (the curv operator) in each iteration. In other terms, this is the strength of the smoothing. This is the parameter µ."
                    },
                    {
                        "name": "Lambda1",
                        "type": "float",
                        "value": 1.,
                        'tip': "MorphGAC algorithm parameter: Relative importance of the inside pixels (lambda1) against the outside pixels (lambda2).",
                        # "suffix": "px",
                        # "siPrefix": False
                    },
                    {
                        "name": "Lambda2",
                        "type": "float",
                        "value": 2.,
                        'tip': "MorphGAC algorithm parameter: Relative importance of the inside pixels (lambda1) against the outside pixels (lambda2).",
                        # "suffix": "px",
                        # "siPrefix": False
                    },
                    {
                        "name": "Iterations",
                        "type": "int",
                        "value": 150
                    },


                ]
            },
            {
                'name': 'Central Vein Segmentation', 'type': 'group',
                'tip': "MorphGAC algorithm parameters. " + _cite,
                'children': [
                {
                    "name": "Smoothing",
                    "type": "float",
                    "value": 2,
                    "suffix": "px",
                    "siPrefix": False,
                    "tip": "MorphGAC algorithm parameter: The number of repetitions of the smoothing step in each iteration. This is the parameter µ."
                },
                {
                    "name": "Threshold",
                    "type": "float",
                    "value": 0.28,
                    'tip': "MorphGAC algorithm parameter: The threshold that determines which areas are affected by the morphological balloon. This is the parameter θ.",
                    # "suffix": "px",
                    # "siPrefix": False
                },
                {
                    "name": "Ballon",
                    "type": "float",
                    "value": -1,
                    'tip': "MorphGAC algorithm parameter: The strength of the morphological balloon. This is the parameter ν.",
                    # "suffix": "px",
                    # "siPrefix": False
                },
                {
                    "name": "Iterations",
                    "type": "int",
                    "value": 400
                },


            ]
            },

        ]

        self.parameters = Parameter.create(name="Lobulus Processing", type="group", children=params)
        self.report: Report = None

    def set_annotated_image_and_id(self, anim: scim.AnnotatedImage, annotation_id, level=None):
        """

        :param anim:
        :param annotation_id:
        :param level: Is used just for short time in funcion get_views()
        :return:
        """
        self.anim = anim
        self.level = level
        self.annotation_id = annotation_id
        self._init_by_annotation_id(annotation_id)

        pass

    def set_report(self, report: Report):
        self.report: Report = report

    def _init_by_annotation_id(self, annotation_id):
        if not np.isscalar(annotation_id):
            raise ValueError("Annotation ID should be scalar int value for lobulus processing.")

        pixelsize_mm = [(self.parameters.param("Working Resolution").value() * 1000)] * 2
        self.view = self.anim.get_views(
            annotation_ids=[annotation_id], level=self.level, margin=1.8,
            pixelsize_mm=pixelsize_mm
        )[0]
        right_shape = imma.image.calculate_new_shape(
            self.view.region_size_on_level,
            self.view.get_pixelsize_on_level(self.view.region_level)[0],
            self.view.region_pixelsize
        )
        self.image = self.view.get_region_image(as_gray=True)
        self.mask = self.view.get_annotation_region_raster(annotation_id=annotation_id)
        pass

    def find_border(self, show=True):
        # inner_lobulus_margin_mm = 0.02

        im_gradient0 = skimage.filters.frangi(self.image)
        im_gradient1 = ms.gborders(self.image, alpha=1000, sigma=2)
        im_gradient = im_gradient1 - (im_gradient0 * 10000)
        # circle = circle_level_set(imgr.shape, size2, 75, scalerow=0.75)
        circle = self.mask
        logger.debug("Image size {}".format(self.image.shape))
        # plt.figure()
        # plt.imshow(im_gradient0)
        # plt.colorbar()
        # plt.contour(circle)
        # plt.show()
        # mgac = ms.MorphGAC(im_gradient, smoothing=2, threshold=0.3, balloon=-1)
        # mgac.levelset = circle.copy()
        # mgac.run(iterations=100)
        # inner = mgac.levelset.copy()

        param_gac_smoothing = self.parameters.param("Central Vein Segmentation", "Smoothing").value()
        param_gac_threshold = self.parameters.param("Central Vein Segmentation", "Threshold").value()
        param_gac_baloon = self.parameters.param("Central Vein Segmentation", "Ballon").value()
        param_gac_iterations = self.parameters.param("Central Vein Segmentation", "Iterations").value()

        param_acwe_smoothing = self.parameters.param("Border Segmentation", "Smoothing").value()
        param_acwe_lambda1 =   self.parameters.param("Border Segmentation", "Lambda1").value()
        param_acwe_lambda2 =   self.parameters.param("Border Segmentation", "Lambda2").value()
        param_acwe_iterations =self.parameters.param("Border Segmentation", "Iterations").value()
        # central vein
        mgac = ms.MorphGAC(im_gradient, smoothing=param_gac_smoothing,
                           threshold=param_gac_threshold, balloon=param_gac_baloon)
        # mgac = ms.MorphGAC(im_gradient, smoothing=2, threshold=0.28, balloon=-1.0)
        # mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=.1, lambda2=.05)
        mgac.levelset = circle.copy()
        mgac.run(iterations=param_gac_iterations)
        inner = mgac.levelset.copy()
        # mgac = ms.MorphGAC(im_gradient, smoothing=2, threshold=0.2, balloon=+1)
        # mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=0.5, lambda2=1.0)

        mgac = ms.MorphACWE(
            im_gradient0,
            smoothing=param_acwe_smoothing,
            lambda1=param_acwe_lambda1,
            lambda2=param_acwe_lambda2)
        mgac.levelset = circle.copy()
        mgac.run(iterations=param_acwe_iterations)
        outer = mgac.levelset.copy()

        # circle = circle_level_set(imgr.shape, (200, 200), 75, scalerow=0.75)

        # plt.figure()
        # plt.imshow(im_gradient0)
        # plt.colorbar()
        # plt.contour(circle + inner + outer)
        # plt.figure()
        # plt.imshow(im_gradient)
        # plt.colorbar()
        # plt.contour(circle + inner + outer)
        plt.figure(figsize=(12, 10))
        plt.imshow(self.image, cmap="gray")
        plt.colorbar()
        plt.contour(circle + inner + outer)
        self.view.add_ticks()

        datarow = {}
        datarow["Annotation ID"] = self.annotation_id
        if self.report is not None:
            plt.savefig(
                op.join(
                    self.report.outputdir, "lobulus_{}.png".format(self.annotation_id)
                )
            )
        if show:
            plt.show()
        self.central_vein_mask = inner
        self.lobulus_mask = (inner + outer) == 1
        datarow["Area"] = np.sum(self.lobulus_mask) * np.prod(
            self.view.region_pixelsize
        )

        datarow["Central vein area"] = np.sum(inner > 0) * np.prod(
            self.view.region_pixelsize
        )
        datarow["Area unit"] = self.view.region_pixelunit
        self.report.add_cols_to_actual_row(datarow)
        self.skeleton_analysis(show=show)

    def skeleton_analysis(self, show=False):
        datarow = {}


        inner = self.central_vein_mask
        # TODO Split the function here
        inner_lobulus_margin_mm = self.parameters.param("Inner Lobulus Margin").value() * 1000

        # eroded image for threshold analysis
        dstmask = scipy.ndimage.morphology.distance_transform_edt(
            self.lobulus_mask, self.view.region_pixelsize
        )
        inner_lobulus_mask = dstmask > inner_lobulus_margin_mm
        # print("inner_lobulus_mask" , np.sum(inner_lobulus_mask==0), np.sum(inner_lobulus_mask==1))

        detail_level = 2
        new_size = self.view.get_size_on_level(detail_level)

        resize_params = dict(
            output_shape=[new_size[1], new_size[0]],
            mode="reflect",
            order=0,
            anti_aliasing=False,
        )
        detail_mask = skimage.transform.resize(self.lobulus_mask, **resize_params)
        detail_inner_lobulus_mask = skimage.transform.resize(
            inner_lobulus_mask, **resize_params
        )
        detail_central_vein_mask = skimage.transform.resize(inner == 1, **resize_params)

        detail_view = self.view.to_level(detail_level)
        detail_image = detail_view.get_region_image(as_gray=True)
        plt.figure()
        plt.imshow(detail_image)
        plt.contour(detail_mask + detail_inner_lobulus_mask)
        detail_view.add_ticks()
        if show:
            plt.show()
        threshold = skimage.filters.threshold_otsu(
            detail_image[detail_inner_lobulus_mask == 1]
        )
        imthr = detail_image < threshold
        imthr[detail_mask != 1] = 0
        # plt.figure()
        # plt.imshow(imthr)
        # if show:
        #     plt.show()
        skeleton = skeletonize(imthr)
        datarow["Skeleton lenght"] = np.sum(skeleton) * detail_view.region_pixelsize[0]
        datarow["Output pixel size 0"] = detail_view.region_pixelsize[0]
        datarow["Output pixel size 1"] = detail_view.region_pixelsize[1]
        datarow["Output image size 0"] = (
            detail_view.region_pixelsize[0] * imthr.shape[0]
        )
        datarow["Output image size 1"] = (
            detail_view.region_pixelsize[1] * imthr.shape[1]
        )
        plt.figure(figsize=(12, 10))
        plt.imshow(skeleton + imthr)
        detail_view.add_ticks()
        if self.report is not None:
            plt.savefig(
                op.join(
                    self.report.outputdir,
                    "thumb_skeleton_thr_{}.png".format(self.annotation_id),
                )
            )
            # skimage.io.imsave(op.join(self.report.outputdir, "figure_skeleton_thumb_{}.png".format(self.annotation_id)), 50 * skeleton + 50 * imthr)
        if show:
            plt.show()

        if self.report is not None:
            imall = detail_mask.astype(np.uint8)
            imall[detail_central_vein_mask > 0] = 2
            imall[imthr > 0] = 3
            imall[skeleton > 0] = 4
            # imall = (skeleton.astype(np.uint8) + imthr.astype(np.uint8) +  + (detail_central_vein_mask.astype(np.uint8) * 2)).astype(np.uint8)
            self.imsave("lobulus_central_thr_skeleton_{}.png", imall)
            self.imsave(
                "lobulus_thr_skeleton_{}.png",
                (skeleton.astype(np.uint8) + imthr + detail_mask).astype(np.uint8),
            )
            self.imsave("skeleton_{}.png", skeleton)
            self.imsave("thr_{}.png", imthr)
            # plt.imsave(op.join(self.report.outputdir, "skeleton_thr_lobulus_{}.png".format(self.annotation_id)), skeleton.astype(np.uint8) + imthr + detail_mask)
            # plt.imsave(op.join(self.report.outputdir, "skeleton_{}.png".format(self.annotation_id)), skeleton)
            # plt.imsave(op.join(self.report.outputdir, "thr_{}.png".format(self.annotation_id)), imthr)
            # skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_thr_lobulus_{}.png".format(self.annotation_id)),
            #                   (50 * skeleton + 50 * imthr + 50 * detail_mask).astype(np.uint8))
            # skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_{}.png".format(self.annotation_id)), 50 * skeleton)
            # skimage.io.imsave(op.join(self.report.outputdir, "raw_thr_{}.png".format(self.annotation_id)), 50 * imthr)

        conv = scipy.signal.convolve2d(skeleton, np.ones([3, 3]), mode="same")
        conv = conv * skeleton
        plt.figure(figsize=(12, 10))
        plt.imshow(conv)
        detail_view.add_ticks()
        if self.report is not None:
            plt.savefig(
                op.join(
                    self.report.outputdir,
                    "figure_skeleton_nodes_{}.png".format(self.annotation_id),
                )
            )

            with warnings.catch_warnings():
                # warnings.simplefilter("low contrast image")
                warnings.filterwarnings("ignore", ".*low contrast image.*")
                self.imsave("skeleton_nodes_{}.png", imthr, 20)
            # plt.imsave(op.join(self.report.outputdir, "skeleton_nodes_{}.png".format(self.annotation_id)), conv.astype(np.uint8))
            # skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_nodes_{}.png".format(self.annotation_id)), (conv * 20).astype(np.uint8))
        if show:
            plt.show()

        conv[conv > 3] = 0
        label, num = scipy.ndimage.label(conv)
        datarow["Branch number"] = num
        label, num = scipy.ndimage.label(conv == 1)
        datarow["Dead ends number"] = num

        self.report.add_cols_to_actual_row(datarow)

    # def imfigsave(self, base_fn, arr):

    def find_cetral_vein(self):
        pass

    def imsave(self, base_fn, arr, k=50):
        base_fn = base_fn.format(self.annotation_id)
        self.report.imsave(base_fn, arr, k)
