# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process lobulus analysis.
"""
import logging
logger = logging.getLogger(__name__)
from scaffan import annotation as scan
from scaffan import image as scim


class Lobulus:
    def __init__(self, anim: scim.AnnotatedImage, annotation_id, level=2):
        self.anim = anim
        self.level = level
        self._init_by_annotation_id(annotation_id)

        pass

    def _init_by_annotation_id(self, annotation_id):
        self.view = self.anim.get_views(annotation_ids=[annotation_id], level=2)[0]
        self.image = self.view.get_region_image(as_gray=True)
        self.mask = self.view.get_annotation_region_raster(annotation_id=annotation_id)
        pass

    def find_border(self):
        pass

    def find_cetral_vein(self):
        pass

