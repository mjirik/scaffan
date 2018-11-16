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
import numpy as np
import io3d
import scaffan
import scaffan.image as scim

import glob
import os

skip_on_local = False

scim.import_openslide()
import openslide
import scaffan.image as scim

class ParseAnnotationTest(unittest.TestCase):

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
        im = anim.get_image_by_center((10000, 10000), as_gray=True )
        self.assertEqual(len(im.shape), 2, "should be 2D")





if __name__ == "__main__":
    # logging.basicConfig(stream=sys.stderr)
    logger.setLevel(logging.DEBUG)
    unittest.main()
