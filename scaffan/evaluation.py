# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Used to support algorithm evaluation. For every inserted annotation it looks for lobulus boundary
and lobulus central vein. The segmentation is compared then and evaluated.
"""

import logging
logger = logging.getLogger(__name__)

import scaffan.image


class Evaluation:
    def __init__(self):
        self.report = None
        pass

    def set_input_data(self, anim: scaffan.image.AnnotatedImage, annotaion_id):

        inner_ids = anim.select_inner_annotations(annotaion_id, color="#000000")
        # TODO add outer annotation
        pass

    def run(self):
        # TODO evaluate
        pass
