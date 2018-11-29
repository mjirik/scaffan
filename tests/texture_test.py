#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
import unittest
import os.path as op
from nose.plugins.attrib import attr

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
        anim.set_region_on_annotations("obj1", 3)
        image = anim.get_region_image()
        plt.imshow(image)
        anim.plot_annotations("obj1")
        # plt.show()
        self.assertGreater(image.shape[0], 100)



if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
