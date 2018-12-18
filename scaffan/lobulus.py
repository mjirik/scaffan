# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process lobulus analysis.
"""
import logging
logger = logging.getLogger(__name__)
import skimage.filters
import morphsnakes as ms
from matplotlib import pyplot as plt
from scaffan import annotation as scan
from scaffan import image as scim


class Lobulus:
    def __init__(self, anim: scim.AnnotatedImage, annotation_id, level=4):
        self.anim = anim
        self.level = level
        self._init_by_annotation_id(annotation_id)

        pass

    def _init_by_annotation_id(self, annotation_id):
        self.view = self.anim.get_views(annotation_ids=[annotation_id], level=self.level, margin=1.5)[0]
        self.image = self.view.get_region_image(as_gray=True)
        self.mask = self.view.get_annotation_region_raster(annotation_id=annotation_id)
        pass

    def find_border(self):
        im_gradient0 = skimage.filters.frangi(self.image)
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

        mgac = ms.MorphGAC(0.00001 -im_gradient0, smoothing=2, threshold=0.7, balloon=-50.0)
        # mgac = ms.MorphACWE(0.0001 - im_gradient0, smoothing=2, lambda1=1.5, lambda2=10.0)
        mgac.levelset = circle.copy()
        mgac.run(iterations=100)
        inner = mgac.levelset.copy()
        # mgac = ms.MorphGAC(im_gradient, smoothing=2, threshold=0.2, balloon=+1)
        # mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=0.5, lambda2=1.0)

        mgac = ms.MorphACWE(im_gradient0, smoothing=2, lambda1=1.5, lambda2=1.0)
        mgac.levelset = circle.copy()
        mgac.run(iterations=10)
        outer = mgac.levelset.copy()

        # circle = circle_level_set(imgr.shape, (200, 200), 75, scalerow=0.75)

        plt.figure(figsize=(15, 10))
        plt.imshow(im_gradient0, cmap="gray")
        plt.colorbar()
        plt.contour(circle + inner + outer)
        plt.show()
        pass

    def find_cetral_vein(self):
        pass

