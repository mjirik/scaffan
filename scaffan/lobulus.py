# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process lobulus analysis.
"""
import logging
logger = logging.getLogger(__name__)
import skimage.filters
import os.path as op
import morphsnakes as ms
from matplotlib import pyplot as plt
from scaffan import annotation as scan
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
        self.view = self.anim.get_views(annotation_ids=[annotation_id], level=self.level, margin=1.5)[0]
        self.image = self.view.get_region_image(as_gray=True)
        self.mask = self.view.get_annotation_region_raster(annotation_id=annotation_id)
        pass

    def find_border(self, show=False):
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

        mgac = ms.MorphGAC(im_gradient, smoothing=2, threshold=0.4, balloon=-1.0)
        # mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=.1, lambda2=.05)
        mgac.levelset = circle.copy()
        mgac.run(iterations=100)
        inner = mgac.levelset.copy()
        # mgac = ms.MorphGAC(im_gradient, smoothing=2, threshold=0.2, balloon=+1)
        # mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=0.5, lambda2=1.0)

        mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=1.0, lambda2=2.0)
        mgac.levelset = circle.copy()
        mgac.run(iterations=150)
        outer = mgac.levelset.copy()

        # circle = circle_level_set(imgr.shape, (200, 200), 75, scalerow=0.75)

        plt.figure()
        plt.imshow(im_gradient0)
        plt.colorbar()
        plt.contour(circle + inner + outer)
        plt.figure()
        plt.imshow(im_gradient)
        plt.colorbar()
        plt.contour(circle + inner + outer)
        plt.figure()
        plt.imshow(self.image, cmap="gray")
        plt.colorbar()
        plt.contour(circle + inner + outer)
        if self.report is not None:
            plt.savefig(op.join(self.report.outputdir, "lobulus_{}.png".format(self.annotation_id)))
        if show:
            plt.show()


    def find_cetral_vein(self):
        pass

