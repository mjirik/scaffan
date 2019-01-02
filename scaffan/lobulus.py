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
import morphsnakes as ms
from matplotlib import pyplot as plt
from scaffan import image as scim


class Lobulus:
    def __init__(self, anim: scim.AnnotatedImage, annotation_id, level=3, report=None):
        self.anim = anim
        self.level = level
        self.report = report
        self.annotation_id = annotation_id
        self._init_by_annotation_id(annotation_id)

        pass

    def _init_by_annotation_id(self, annotation_id):
        self.view = self.anim.get_views(annotation_ids=[annotation_id], level=self.level, margin=1.8)[0]
        self.image = self.view.get_region_image(as_gray=True)
        self.mask = self.view.get_annotation_region_raster(annotation_id=annotation_id)
        pass

    def find_border(self, show=True):
        inner_lobulus_margin_mm = 0.02

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

        # central vein
        mgac = ms.MorphGAC(im_gradient, smoothing=2, threshold=0.28, balloon=-1.0)
        # mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=.1, lambda2=.05)
        mgac.levelset = circle.copy()
        mgac.run(iterations=150)
        inner = mgac.levelset.copy()
        # mgac = ms.MorphGAC(im_gradient, smoothing=2, threshold=0.2, balloon=+1)
        # mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=0.5, lambda2=1.0)

        mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=1.0, lambda2=2.0)
        mgac.levelset = circle.copy()
        mgac.run(iterations=400)
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
            plt.savefig(op.join(self.report.outputdir, "lobulus_{}.png".format(self.annotation_id)))
        if show:
            plt.show()
        self.lobulus_mask = (inner + outer) == 1
        datarow["Area"] = np.sum(self.lobulus_mask) * np.prod(self.view.region_pixelsize)
        datarow["Area unit"] = self.view.region_pixelunit

        # eroded image for threshold analysis
        dstmask = scipy.ndimage.morphology.distance_transform_edt(self.lobulus_mask, self.view.region_pixelsize)
        inner_lobulus_mask = (dstmask > inner_lobulus_margin_mm)
        # print("inner_lobulus_mask" , np.sum(inner_lobulus_mask==0), np.sum(inner_lobulus_mask==1))

        detail_level = 2
        new_size = self.view.get_size_on_level(detail_level)

        detail_mask = skimage.transform.resize(self.lobulus_mask, [new_size[1], new_size[0]], mode="reflect", order=0)
        detail_inner_lobulus_mask = skimage.transform.resize(inner_lobulus_mask, [new_size[1], new_size[0]], mode="reflect", order=0)
        detail_central_vein_mask = skimage.transform.resize(inner == 1, [new_size[1], new_size[0]], mode="reflect", order=0)

        detail_view = self.view.to_level(detail_level)
        detail_image = detail_view.get_region_image(as_gray=True)
        plt.figure()
        plt.imshow(detail_image)
        plt.contour(detail_mask + detail_inner_lobulus_mask)
        detail_view.add_ticks()
        if show:
            plt.show()
        threshold = skimage.filters.threshold_otsu(detail_image[detail_inner_lobulus_mask == 1])
        imthr = (detail_image < threshold)
        imthr[detail_mask != 1] = 0
        # plt.figure()
        # plt.imshow(imthr)
        # if show:
        #     plt.show()
        skeleton = skeletonize(imthr)
        datarow["Skeleton lenght"] = np.sum(skeleton) * detail_view.region_pixelsize[0]
        datarow["Output pixel size 0"] = detail_view.region_pixelsize[0]
        datarow["Output pixel size 1"] = detail_view.region_pixelsize[1]
        datarow["Output image size 0"] = detail_view.region_pixelsize[0] * imthr.shape[0]
        datarow["Output image size 1"] = detail_view.region_pixelsize[1] * imthr.shape[1]
        plt.figure(figsize=(12, 10))
        plt.imshow(skeleton + imthr)
        detail_view.add_ticks()
        if self.report is not None:
            plt.savefig(op.join(self.report.outputdir, "thumb_skeleton_thr_{}.png".format(self.annotation_id)))
            # skimage.io.imsave(op.join(self.report.outputdir, "figure_skeleton_thumb_{}.png".format(self.annotation_id)), 50 * skeleton + 50 * imthr)
        if show:
            plt.show()

        if self.report is not None:
            imall = (skeleton.astype(np.uint8) + imthr.astype(np.uint8) + detail_mask.astype(np.uint8) * 2 + (detail_central_vein_mask.astype(np.uint8))).astype(np.uint8)
            self.imsave("skeleton_lobulus_thr_central_{}.png", imall)
            plt.imsave(op.join(self.report.outputdir, "skeleton_thr_lobulus_{}.png".format(self.annotation_id)), skeleton.astype(np.uint8) + imthr + detail_mask)
            plt.imsave(op.join(self.report.outputdir, "skeleton_{}.png".format(self.annotation_id)), skeleton)
            plt.imsave(op.join(self.report.outputdir, "thr_{}.png".format(self.annotation_id)), imthr)
            skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_thr_lobulus_{}.png".format(self.annotation_id)),
                              (50 * skeleton + 50 * imthr + 50 * detail_mask).astype(np.uint8))
            skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_{}.png".format(self.annotation_id)), 50 * skeleton)
            skimage.io.imsave(op.join(self.report.outputdir, "raw_thr_{}.png".format(self.annotation_id)), 50 * imthr)

        conv = scipy.signal.convolve2d(skeleton, np.ones([3, 3]), mode="same")
        conv = conv * skeleton
        plt.figure(figsize=(12, 10))
        plt.imshow(conv)
        detail_view.add_ticks()
        if self.report is not None:
            plt.savefig(op.join(self.report.outputdir, "figure_skeleton_nodes_{}.png".format(self.annotation_id)))
            plt.imsave(op.join(self.report.outputdir, "skeleton_nodes_{}.png".format(self.annotation_id)), conv.astype(np.uint8))
            skimage.io.imsave(op.join(self.report.outputdir, "raw_skeleton_nodes_{}.png".format(self.annotation_id)), (conv * 20).astype(np.uint8))
        if show:
            plt.show()

        conv[conv > 3] = 0
        label, num = scipy.ndimage.label(conv, )
        datarow["Branch number"] = num

        self.report.add_row(datarow)

    # def imfigsave(self, base_fn, arr):

    def imsave(self, base_fn, arr, k=50):
        """

        :param base_fn: with a format slot for annotation id like "skeleton_{}.png"
        :param arr:
        :return:
        """
        plt.imsave(op.join(self.report.outputdir, base_fn.format(self.annotation_id)), arr)
        skimage.io.imsave(op.join(self.report.outputdir, "raw_" + base_fn.format(self.annotation_id)), k * arr)

    def find_cetral_vein(self):
        pass

