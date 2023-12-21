import shutil

from loguru import logger
import unittest
import os
import os.path as op
import sys
import io3d
from pathlib import Path
from unittest.mock import patch

path_to_script = op.dirname(op.abspath(__file__))
# sys.path.insert(0, op.abspath(op.join(path_to_script, "../../exsu")))
# # sys.path.insert(0, op.abspath(op.join(path_to_script, "../../imma")))
# exsu_pth = Path(__file__).parents[2] / "exsu"
# logger.debug(f"exsupth{exsu_pth}, {exsu_pth.exists()}")
# sys.path.insert(0, exsu_pth)

import exsu

logger.debug(f"exsu path: {exsu.__file__}")
# import openslide
import scaffan
import scaffan.algorithm
import scaffan.skeleton_analysis

# import scaffan
import scaffan.image
import skimage.io
from PyQt5 import QtWidgets
import pytest
from datetime import datetime





def test_run_lobulus_and_skeleton_with_seeds_mm():
    """
    Try to run by seeds mm. Just few iterations.
    :return:
    """
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    mainapp = scaffan.algorithm.Scaffan()
    mainapp.set_output_dir(".test_run_skeleton_with_seeds_mm")
    mainapp.set_parameter("Processing;Whole Scan Segmentation", False)
    mainapp.set_parameter("Input;Lobulus Selection Method", "Auto") # this is not evaluated
    mainapp.set_parameter("Processing;Skeleton Analysis", True)
    mainapp.set_parameter("Processing;Texture Analysis", False)
    mainapp.set_parameter("Processing;Open output dir", False)
    mainapp.set_parameter("Processing;SNI Prediction CNN", False)
    mainapp.set_parameter(
        "Processing;Lobulus Segmentation;Border Segmentation;Iterations", 10
    )
    mainapp.set_parameter(
        "Processing;Lobulus Segmentation;Central Vein Segmentation;Iterations", 10
    )
    mainapp.set_input_file(fn)
    mainapp.run_lobuluses(seeds_mm=[[6.86, 6.86]])
    assert (0.01 < mainapp.report.df["Area"][0]), "At least something should be found"

def test_run_lobulus_and_skeleton_with_seeds_mm():
    """
    Try to run by seeds mm. Just few iterations.
    :return:
    """
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )

    fn = io3d.joinp("medical/orig/scaffan_png_tiff/split_176_landscape.tif")
    img = skimage.io.imread(fn, as_gray=True)
    import exsu.report
    report = exsu.report.Report(".odir")
    data = scaffan.skeleton_analysis._thresholding_and_skeletonization(img, report=report)
    assert "Dead ends number" in data


