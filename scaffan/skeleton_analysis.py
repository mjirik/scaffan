# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process lobulus analysis.
"""

from loguru import logger
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
import scaffan
import scaffan.lobulus
from exsu.report import Report
from pyqtgraph.parametertree import Parameter
import imma.image


class SkeletonAnalysis:
    def __init__(
        self,
        pname="Skeleton Analysis",
        ptype="bool",
        pvalue=True,
        ptip="Skeleton Analysis after lobulus segmentation is performed",
    ):

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
                "value": 0.00000091,  # this is typical resolution on level 2
                # "value": 0.00000182,  # this is typical resolution on level 3
                # "value": 0.00000364,  # this is typical resolution on level 4
                # "value": 0.00000728,  # this is typical resolution on level 5
                # "value": 0.00001456,  # this is typical resolution on level 6
                "suffix": "m",
                "siPrefix": True,
            },
            {
                "name": "Inner Lobulus Margin",
                "type": "float",
                "value": 0.00002,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Area close to the border is ignored in Otsu threshold computation before skeletonization step.",
            },
        ]

        self.parameters = Parameter.create(
            name=pname,
            type=ptype,
            value=pvalue,
            tip=ptip,
            children=params,
            expanded=False,
        )
        self.report: Report = None

    def set_report(self, report: Report):
        self.report = report

    def set_lobulus(self, lobulus: scaffan.lobulus.Lobulus):
        self.lobulus: scaffan.lobulus.Lobulus = lobulus
        pixelsize_mm = [
            (self.parameters.param("Working Resolution").value() * 1000)
        ] * 2
        self.view = self.lobulus.view.to_pixelsize(pixelsize_mm=pixelsize_mm)
        logger.debug("Lobulus setup done")

    def skeleton_analysis(self, show=False):
        datarow = {}

        inner = self.lobulus.central_vein_mask
        logger.debug(
            f"lobulus.central_vein_mask: dtype={self.lobulus.central_vein_mask.dtype}, "
            + f"shape={self.lobulus.central_vein_mask.shape}, "
            + f"unique={np.unique(self.lobulus.central_vein_mask)}"
        )
        # import pdb; pdb.set_trace()
        # TODO Split the function here
        inner_lobulus_margin_mm = (
            self.parameters.param("Inner Lobulus Margin").value() * 1000
        )

        logger.debug(
            f"Distance transform. type(mask)={str(type(self.lobulus.lobulus_mask))}, "
            f"mask.shape={self.lobulus.lobulus_mask.shape}, "
            f"mask.unique={np.unique(self.lobulus.lobulus_mask, return_counts=True)}"
        )
        # eroded image for threshold analysis
        dstmask = scipy.ndimage.morphology.distance_transform_edt(
            self.lobulus.lobulus_mask, self.lobulus.view.region_pixelsize[::-1]
        )
        inner_lobulus_mask = dstmask > inner_lobulus_margin_mm
        logger.debug(
            f"inner_lobulus_mask: unique/counts={np.unique(inner_lobulus_mask, return_counts=True)}"
        )

        # detail_level = 2
        new_size = self.view.get_size_on_pixelsize_mm()

        resize_params = dict(
            output_shape=[new_size[1], new_size[0]],
            mode="reflect",
            order=0,
            anti_aliasing=False,
        )
        logger.debug(
            f"Resizing mask from {self.lobulus.lobulus_mask.shape} to {resize_params['output_shape']}"
        )
        detail_mask = skimage.transform.resize(
            self.lobulus.lobulus_mask, **resize_params
        ).astype(np.int8)
        logger.debug(
            f"Resizing mask from {inner_lobulus_mask.shape} to {resize_params['output_shape']}"
        )
        detail_inner_lobulus_mask = skimage.transform.resize(
            inner_lobulus_mask, **resize_params
        )
        logger.debug(
            f"Resizing mask from {inner.shape} to {resize_params['output_shape']}"
        )
        detail_central_vein_mask = skimage.transform.resize(
            inner == 1, **resize_params
        ).astype(np.int8)

        logger.debug(f"Preparing to show")
        detail_view = self.view
        logger.debug(f"view={detail_view}")
        # TODO change log_level to trace
        detail_image = detail_view.get_region_image(as_gray=True, log_level="DEBUG")
        logger.debug("preparing figure")
        fig = plt.figure()
        plt.imshow(detail_image)
        plt.contour(detail_mask + detail_inner_lobulus_mask)
        detail_view.add_ticks()
        logger.debug("fig to report...")
        if self.report is not None:
            self.report.savefig_and_show(
                "skeleton_analysis_detail_image_and_mask_{}.png".format(
                    self.lobulus.annotation_id
                ),
                fig,
            )

        # from PyQt5.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()

        # Or for Qt5
        # from PyQt5.QtCore import pyqtRemoveInputHook

        # from pdb import set_trace
        # set_trace()
        logger.debug(
            f"detail_inner_lobulus_mask {detail_central_vein_mask.shape} {detail_central_vein_mask.dtype} "
            f"{np.min(detail_central_vein_mask)} {np.max(detail_central_vein_mask)}"
        )
        logger.debug("Thresholding and skeletonization...")
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
        sumskel = np.sum(skeleton)
        logger.debug(
            f"Skeletonization finished. threshold={threshold}, sumskel={sumskel}"
        )
        datarow["Skeleton length"] = sumskel * detail_view.region_pixelsize[0]
        datarow["Output pixel size 0"] = detail_view.region_pixelsize[0]
        datarow["Output pixel size 1"] = detail_view.region_pixelsize[1]
        datarow["Output image size 0"] = (
            detail_view.region_pixelsize[0] * imthr.shape[0]
        )
        datarow["Output image size 1"] = (
            detail_view.region_pixelsize[1] * imthr.shape[1]
        )
        fig = plt.figure(figsize=(12, 10))
        plt.imshow(skeleton + imthr)
        detail_view.add_ticks()
        if self.report is not None:
            self.report.savefig_and_show(
                "thumb_skeleton_thr_{}.png".format(self.lobulus.annotation_id), fig
            )

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
                severity=55,
            )
            self.imsave("skeleton_{}.png", skeleton, 55)
            self.imsave("thr_{}.png", imthr)
            # plt.imsave(op.join(self.report.outputdir, "skeleton_thr_lobulus_{}.png".format(self.annotation_id)), skeleton.astype(np.uint8) + imthr + detail_mask)
            # plt.imsave(op.join(self.report.outputdir, "skeleton_{}.png".format(self.annotation_id)), skeleton)
            # plt.imsave(op.join(self.report.outputdir, "thr_{}.png".format(self.annotation_id)), imthr)
            # skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_thr_lobulus_{}.png".format(self.annotation_id)),
            #                   (50 * skeleton + 50 * imthr + 50 * detail_mask).astype(np.uint8))
            # skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_{}.png".format(self.annotation_id)), 50 * skeleton)
            # skimage.io.imsave(op.join(self.report.outputdir, "raw_thr_{}.png".format(self.annotation_id)), 50 * imthr)

        logger.debug("Branching points detection...")
        conv = scipy.signal.convolve2d(skeleton, np.ones([3, 3]), mode="same")
        conv = conv * skeleton
        fig = plt.figure(figsize=(12, 10))
        plt.imshow(conv)
        detail_view.add_ticks()
        if self.report is not None:
            self.report.savefig_and_show(
                "figure_skeleton_nodes_{}.png".format(self.lobulus.annotation_id), fig
            )

            with warnings.catch_warnings():
                # warnings.simplefilter("low contrast image")
                warnings.filterwarnings("ignore", ".*low contrast image.*")
                self.imsave("skeleton_nodes_raw_{}.png", conv, 20)
            # plt.imsave(op.join(self.report.outputdir, "skeleton_nodes_{}.png".format(self.annotation_id)), conv.astype(np.uint8))
            # skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_nodes_{}.png".format(self.annotation_id)), (conv * 20).astype(np.uint8))

        conv[conv > 3] = 0
        label, num = scipy.ndimage.label(conv)
        datarow["Branch number"] = num
        label, num = scipy.ndimage.label(conv == 1)
        datarow["Dead ends number"] = num
        if "Area" in datarow:
            area_unit = datarow["Area unit"]
            datarow[f"Branch number density [1/{area_unit}^2]"] = (
                datarow["Branch number"] / datarow["Area"]
            )
            datarow[f"Dead ends number density [1/{area_unit}^2]"] = (
                datarow["Dead ends number"] / datarow["Area"]
            )
            datarow[f"Skeleton length density [{area_unit}/{area_unit}^2]"] = (
                datarow["Branch length number"] / datarow["Area"]
            )
        else:
            # probably area can be estimated by view area
            logger.debug("Unknown area. Skipping density calculation")

        if "Lobulus Equivalent Surface" in datarow:
            area_unit = datarow["Area unit"]
            datarow[f"Equivalent branch number density [1/{area_unit}^2]"] = (
                datarow["Branch number"] / datarow["Lobulus Equivalent Surface"]
            )
            datarow[f"Equivalent dead ends number density [1/{area_unit}^2]"] = (
                datarow["Dead ends number"] / datarow["Lobulus Equivalent Surface"]
            )
            datarow[
                f"Equivalent skeleton length density [{area_unit}/{area_unit}^2]"
            ] = (
                datarow["Branch length number"] / datarow["Lobulus Equivalent Surface"]
            )
        else:
            # probably area can be estimated by view area
            logger.debug(
                "Unknown 'Lobulus Equivalent Surface'. Skipping density calculation"
            )

        self.report.add_cols_to_actual_row(datarow)
        logger.debug("Skeleton analysis finished.")

    def imsave(self, base_fn, arr, severity=50):
        base_fn = base_fn.format(self.lobulus.annotation_id)
        self.report.imsave(base_fn, arr, severity)
