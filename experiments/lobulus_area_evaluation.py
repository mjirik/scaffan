#! /usr/bin/python
# -*- coding: utf-8 -*-

from loguru import logger
import unittest
import os
import io3d
import os.path as op
import sys
from pathlib import Path
import datetime

experiment_datetime = datetime.datetime.now()
experiment_datetime_fn = experiment_datetime.strftime("%Y%m%d-%H%M%S")
experiment_dir = Path(f"SA_{experiment_datetime_fn}_area")

experiment_dir.mkdir()
# logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
# logger.add(sys.stderr, format="{time} {level} {message}", level="DEBUG")
logger.add(experiment_dir/"scaffan.log", format="{time} {level} {message}",  level="DEBUG", backtrace=True)
logger.debug("logging init")

path_to_script = os.path.dirname(os.path.abspath(__file__))
path_to_scaffan = os.path.join(path_to_script, "..")
# print("insert path: ", path_to_scaffan)

sys.path.insert(0, path_to_scaffan)
import scaffan
import scaffan.algorithm
# fn = io3d.datasets.join_path(
#     "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
# )
# fn = io3d.datasets.join_path(
#     "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-4_HE_parenchyme.ndpi", get_root=True
# )
# logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
    # .isoformat(' ', 'seconds')
# datetime.datetime.now().
experiment_title = "first area evaluation"
logger.info(f"running experiment: {experiment_title} started at: {experiment_datetime}")
# imsl = openslide.OpenSlide(fn)
# annotations = scan.read_annotations(fn)
# scan.annotations_to_px(imsl, annotations)
mainapp = scaffan.algorithm.Scaffan()


#############
# mainapp.set_output_dir(experiment_dir/"PIG-001")

mainapp.set_persistent_cols({
    "Experiment Title": experiment_title,
    "Experiment Datetime": experiment_datetime.isoformat(" ", "seconds"),
})

mainapp.set_parameter("Processing;Lobulus Segmentation;Central Vein Segmentation;Threshold", 0.18)
mainapp.set_parameter("Processing;Run Skeleton Analysis", True)
mainapp.set_parameter("Processing;Run Texture Analysis", True)
# mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", False)

# mainapp.set_parameter("Processing;Run Skeleton Analysis", False)
# mainapp.set_parameter("Processing;Run Texture Analysis", False)
mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", True)

def set_same(mainapp, fn):
    logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
    mainapp.set_input_file(fn)
    odir = experiment_dir / Path(fn).stem
    mainapp.set_output_dir(odir)
    logger.debug(f"output dir: {str(odir)}")
    # mainapp.set_annotation_color_selection("#0000FF") # Blue is used for unlabeled
    mainapp.set_annotation_color_selection("#00FF00")
    # mainapp.parameters.param("Processing", "Lobulus Segmentation", "Central Vein Segmentation", "Threshold").setValue(0.20)
    # mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", True)

fns = [
    "medical/orig/Scaffan-analysis/PIG-001_J-17-0571_LM central_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-002_J-18-0091_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-002_J-18-0092_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-003_J-18-0165_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-003_J-18-0166_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-4_HE_parenchyme.ndpi",
    "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-2 _HE_parenchyme.ndpi"

]

# P001
for fn in fns:
    set_same(mainapp, io3d.datasets.join_path(fn, get_root=True))
    mainapp.run_lobuluses(None)

# # P002
# fn = io3d.datasets.join_path(
#     , get_root=True
# )
# set_same(mainapp, fn)
# mainapp.run_lobuluses(None)
#
# fn = io3d.datasets.join_path(
#     , get_root=True
# )
# set_same(mainapp, fn)
# mainapp.run_lobuluses(None)
#
#
# #############
#
# # mainapp.set_output_dir(experiment_dir/"PIG-003")
# fn = io3d.datasets.join_path(
#     , get_root=True
# )
# set_same(mainapp, fn)
# # logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
# # mainapp.set_input_file(fn)
# # mainapp.set_annotation_color_selection("#00FF00")
# mainapp.run_lobuluses(None)
#
# fn = io3d.datasets.join_path(
#     , get_root=True
# )
# set_same(mainapp, fn)
# mainapp.run_lobuluses(None)
# #############
#
#
#
# # mainapp.set_output_dir(experiment_dir/"PIG-004")
# fn = io3d.datasets.join_path(
#     , get_root=True
# )
# # set_same(mainapp, fn)
# # logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
# # mainapp.set_input_file(fn)
# # mainapp.set_annotation_color_selection("#00FF00")
# mainapp.run_lobuluses(None)
#
# fn = io3d.datasets.join_path(
#     , get_root=True
# )
# set_same(mainapp, fn)
# mainapp.run_lobuluses(None)
