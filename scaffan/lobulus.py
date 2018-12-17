# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process lobulus analysis.
"""
import logging
logger = logging.getLogger(__name__)
from scaffan import annotation as scan
from scaffan import annotation as scan


class Lobulus:
    def __init__(self, anim, annotation_id):
        self.anim = anim
        self._init_by_annotation_id(annotation_id)

        pass

    def _init_by_annotation_id(self, annotation_id):
        pass

