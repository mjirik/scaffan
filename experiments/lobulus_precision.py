#! /usr/bin/python
# -*- coding: utf-8 -*-

from loguru import logger
import unittest
import os
import io3d
import os.path as op
import sys
from pathlib import Path

# logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
# logger.add(sys.stderr, format="{time} {level} {message}", level="DEBUG")
logger.add("scaffan.log", format="{time} {level} {message}",  level="DEBUG")
logger.debug("logging init")

path_to_script = os.path.dirname(os.path.abspath(__file__))
path_to_scaffan = os.path.join(path_to_script, "..")
# print("insert path: ", path_to_scaffan)

sys.path.insert(0, path_to_scaffan)
import scaffan
import scaffan.algorithm
fn = io3d.datasets.join_path(
    "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
)
fn = io3d.datasets.join_path(
    "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-4_HE_parenchyme.ndpi", get_root=True
)
logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
# imsl = openslide.OpenSlide(fn)
# annotations = scan.read_annotations(fn)
# scan.annotations_to_px(imsl, annotations)
mainapp = scaffan.algorithm.Scaffan()
mainapp.set_output_dir("test_run_lobuluses_output_dir")
# mainapp.init_run()
# mainapp.set_annotation_color_selection("#FF00FF")

fn = io3d.datasets.join_path(
    "medical/orig/Scaffan-analysis/PIG-001_J-17-0571_LM central_HE.ndpi", get_root=True
)
logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
mainapp.set_input_file(fn)
mainapp.set_annotation_color_selection("#00FF00")
mainapp.run_lobuluses(None)


# fn = io3d.datasets.join_path(
#     "medical/orig/Scaffan-analysis/PIG-003_J-18-0165_HE.ndpi", get_root=True
# )
# logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
# mainapp.set_input_file(fn)
# mainapp.set_annotation_color_selection("#0000FF")
# mainapp.run_lobuluses(None)
#
#
#
#
#
# fn = io3d.datasets.join_path(
#     "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-4_HE_parenchyme.ndpi", get_root=True
# )
# logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
# mainapp.set_input_file(fn)
# mainapp.set_annotation_color_selection("#00FF00")
# mainapp.run_lobuluses(None)
