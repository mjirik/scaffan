#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Train new parameters for HCTF segmentation
"""

from loguru import logger
import unittest
import os
import io3d
import io3d.datasets
import os.path as op
import sys
from pathlib import Path
import datetime

experiment_title = "Whole Scan Segmentation training"
method = "GLCMTFS"

experiment_datetime = datetime.datetime.now()
experiment_datetime_fn = experiment_datetime.strftime("%Y%m%d-%H%M%S")
experiment_dir = Path(
    io3d.datasets.join_path(
        f"medical/processed/SA_{experiment_datetime_fn}_slide_segmentation_training",
        get_root=True,
    )
)

experiment_dir.mkdir()
# logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
# logger.add(sys.stderr, format="{time} {level} {message}", level="DEBUG")
logger.add(
    experiment_dir / "scaffan.log",
    format="{time} {level} {message}",
    level="DEBUG",
    backtrace=True,
)
logger.debug("logging init")
logger.debug(f"experiment dir {experiment_dir}")

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
logger.info(f"running experiment: {experiment_title} started at: {experiment_datetime}")
# imsl = openslide.OpenSlide(fn)
# annotations = scan.read_annotations(fn)
# scan.annotations_to_px(imsl, annotations)
mainapp = scaffan.algorithm.Scaffan(default_output_dir_prefix=experiment_dir)
mainapp.set_parameter("Processing;Whole Scan Segmentation;Segmentation Method", method)
mainapp.slide_segmentation.init_clf()  # this is usually automatically called in run()
clf_fn = None  # rewrite the original
if clf_fn is not None:
    mainapp.slide_segmentation.clf_fn = clf_fn
clf_fn = Path(mainapp.slide_segmentation.clf_fn)
if clf_fn.exists():
    modtime0 = datetime.datetime.fromtimestamp(clf_fn.stat().st_mtime)
else:
    modtime0 = ""
logger.debug(f"classificator prior modification time: {modtime0}")

#############
# mainapp.set_output_dir(experiment_dir/"PIG-001")

mainapp.set_persistent_cols(
    {
        "Experiment Title": experiment_title,
        "Experiment Datetime": experiment_datetime.isoformat(" ", "seconds"),
    }
)

mainapp.set_parameter("Processing;Intensity Normalization", False)
# mainapp.set_parameter("Processing;Intensity Normalization", True)
# mainapp.set_parameter("Processing;Whole Scan Segmentation;Working Resolution", 0.000005) # default is 0.000001 (10 um)
# mainapp.set_parameter("Processing;Whole Scan Segmentation;Working Resolution", 0.0000025) # default is 0.000001 (10 um)
# mainapp.set_parameter("Processing;Lobulus Segmentation;Central Vein Segmentation;Threshold", 0.18)
# mainapp.set_parameter("Processing;Lobulus Segmentation;Central Vein Segmentation;Threshold", 0.20)
# mainapp.parameters.param("Processing", "Lobulus Segmentation", "Central Vein Segmentation", "Threshold").setValue(0.20)
# mainapp.set_parameter("Processing;Skeleton Analysis", True)
# mainapp.set_parameter("Processing;Texture Analysis", True)
# mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", False)

# mainapp.set_parameter("Processing;Skeleton Analysis", False)
# mainapp.set_parameter("Processing;Texture Analysis", False)
mainapp.set_parameter("Processing;Whole Scan Segmentation;Save Training Labels", True)
mainapp.set_report_level(10)
# mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", True)

# fns = [
#     "medical/orig/Scaffan-analysis/PIG-001_J-17-0571_LM central_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-001_J-17-0569_LM_HE.ndpi",
#     # "medical/orig/Scaffan-analysis/PIG-001_J-17-0567_edge RM_HE.ndpi",  # no annotation
#     "medical/orig/Scaffan-analysis/PIG-002_J-18-0091_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-002_J-18-0092_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-002_J-18-0094_HE_rescan.ndpi", # bad focus
#     "medical/orig/Scaffan-analysis/PIG-003_J-18-0165_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-003_J-18-0166_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-003_J-18-0167_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-003_J-18-0168_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-003_J-18-0169_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-003_J-18-0170_HE.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-4_HE_parenchyme.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-3 _HE_parenchyme.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-2 _HE_parenchyme.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-005_J-18-0633_HE_PRML per decell.ndpi",
#     "medical/orig/Scaffan-analysis/PIG-008_P008 LL-C_HE_parenchyme centr..ndpi",
#     "medical/orig/Scaffan-analysis/PIG-008_P008 LL-P_HE_parenchyme perif..ndpi",
#
# ]
fns = [
    # number of areas: 9, 17, 4,
    #     io3d.datasets.join_path("medical", "orig", "Scaffan-analysis", "PIG-002_J-18-0091_HE.ndpi", get_root=True), # ugly data
    # training
    # io3d.datasets.join_path("medical", "orig", "Scaffan-analysis", "PIG-003_J-18-0165_HE.ndpi", get_root=True), # test data
    # training
    io3d.datasets.join_path(
        "medical",
        "orig",
        "Scaffan-analysis",
        "PIG-003_J-18-0168_HE.ndpi",
        get_root=True,
    ),
    # training
    # number of areas: 3, 6, 5, 9, 7
    io3d.datasets.join_path(
        "medical",
        "orig",
        "Scaffan-analysis",
        "PIG-003_J-18-0169_HE.ndpi",
        get_root=True,
    ),  # training  bubles
    io3d.datasets.join_path(
        "medical",
        "orig",
        "Scaffan-analysis",
        "PIG-005_J-18-0631_HE_LML per decell.ndpi",
        get_root=True,
    ),
    io3d.datasets.join_path(
        "medical",
        "orig",
        "Scaffan-analysis",
        "PIG-005_J-18-0633_HE_PRML per decell.ndpi",
        get_root=True,
    ),
    io3d.datasets.join_path(
        "medical",
        "orig",
        "Scaffan-analysis",
        "Pig-009_Pilsen_HE_009-LM-P.ndpi",
        get_root=True,
    ),
    io3d.datasets.join_path(
        "medical",
        "orig",
        "Scaffan-analysis",
        "PIG-004_BBJ-004-3 _HE_parenchyme.ndpi",
        get_root=True,
    ),
    io3d.datasets.joinp(
        "medical", "orig", "scaner_zeiss", "Moulisova-JENA", "06_2019_11_12__-1-3.czi"
    ),
    io3d.datasets.joinp(
        "medical", "orig", "scaner_zeiss", "Moulisova-JENA", "07_2019_11_12__f-1.czi"
    ),
    io3d.datasets.joinp(
        "medical", "orig", "scaner_zeiss", "Moulisova-JENA", "07_2019_11_13__l-2.czi"
    ),
    io3d.datasets.joinp(
        "medical", "orig", "scaner_zeiss", "Moulisova-JENA", "08_2019_11_12__-1-4.czi"
    ),
    io3d.datasets.joinp(
        "medical", "orig", "scaner_zeiss", "Moulisova-JENA", "08_2019_11_13__-4.czi"
    ),
    # io3d.datasets.joinp("medical", "orig", "scaner_zeiss", "Moulisova-JENA", "11_2019_11_13__-7.czi"), # test data
    #     io3d.datasets.joinp("medical", "orig", "scaner_zeiss", "Moulisova-JENA", "")
]

mainapp.train_scan_segmentation(fns)
# for fn in fns:
#     set_same(mainapp, io3d.datasets.join_path(fn, get_root=True))
#     mainapp.run_lobuluses(None)
