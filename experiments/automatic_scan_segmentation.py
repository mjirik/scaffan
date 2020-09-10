#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Segmentation is done with training update to actual data
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

experiment_title = "scan segmentation default and each training"
experiment_title = "scan segmentation default and each training with weight"
method = "GLCMTFS"

experiment_datetime = datetime.datetime.now()
experiment_datetime_fn = experiment_datetime.strftime("%Y%m%d-%H%M%S")
experiment_dir = Path(io3d.datasets.join_path(f"medical/processed/SA_{experiment_datetime_fn}_slide_segmentation", get_root=True))

experiment_dir.mkdir()
# logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
# logger.add(sys.stderr, format="{time} {level} {message}", level="DEBUG")
logger.add(experiment_dir/"scaffan.log", format="{time} {level} {message}",  level="DEBUG", backtrace=True)
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
mainapp = scaffan.algorithm.Scaffan()


#############
# mainapp.set_output_dir(experiment_dir/"PIG-001")

mainapp.set_persistent_cols({
    "Experiment Title": experiment_title,
    "Experiment Datetime": experiment_datetime.isoformat(" ", "seconds"),
})

mainapp.set_parameter("Processing;Scan Segmentation;Segmentation Method", method)
mainapp.set_parameter("Processing;Intensity Normalization", False)
# mainapp.set_parameter("Processing;Intensity Normalization;Run Intensity Normalization", True)
# mainapp.set_parameter("Processing;Scan Segmentation;Working Resolution", 0.000005) # default is 0.000001 (10 um)
mainapp.set_parameter("Processing;Scan Segmentation;Working Resolution", 0.0000025) # default is 0.000001 (10 um)
# mainapp.set_parameter("Processing;Lobulus Segmentation;Central Vein Segmentation;Threshold", 0.18)
mainapp.set_parameter("Processing;Lobulus Segmentation;Central Vein Segmentation;Threshold", 0.20)
# mainapp.parameters.param("Processing", "Lobulus Segmentation", "Central Vein Segmentation", "Threshold").setValue(0.20)
# mainapp.set_parameter("Processing;Skeleton Analysis", True)
# mainapp.set_parameter("Processing;Texture Analysis", True)
mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", False)

mainapp.set_parameter("Processing;Skeleton Analysis", False)
mainapp.set_parameter("Processing;Texture Analysis", False)
mainapp.set_report_level(35)
# mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", True)

def set_same(mainapp, fn):
    logger.debug(f"fn exists {Path(fn).exists()}, fn: {fn}")
    mainapp.set_input_file(fn)
    odir = experiment_dir / Path(fn).stem
    mainapp.set_output_dir(odir)
    logger.debug(f"output dir: {str(odir)}")
    # mainapp.set_annotation_color_selection("#0000FF") # Blue is used for unlabeled
    # mainapp.set_annotation_color_selection("#00FF00")
    # mainapp.set_annotation_color_selection("#FFFF00")
    mainapp.set_parameter("Input;Lobulus Selection Method", "Auto")
    mainapp.set_parameter("Processing;Skeleton Analysis", False)
    mainapp.set_parameter("Processing;Texture Analysis", False)
    mainapp.set_parameter("Processing;Open output dir", False)
    mainapp.set_parameter("Processing;Scan Segmentation;TFS General;Load Default Classifier", True)
    mainapp.set_parameter("Processing;Scan Segmentation;TFS General;Clean Before Training", False)
    mainapp.set_parameter("Processing;Scan Segmentation;TFS General;Training Weight", 100)
    mainapp.set_parameter("Processing;Scan Segmentation;TFS General;Run Training", True)
    mainapp.set_parameter("Processing;Scan Segmentation;Lobulus Number", 0)
    # mainapp.set_parameter("Processing;Lobulus Segmentation;Manual Segmentation", True)

fns = [
    "medical/orig/Scaffan-analysis/PIG-001_J-17-0571_LM central_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-001_J-17-0569_LM_HE.ndpi",
    # "medical/orig/Scaffan-analysis/PIG-001_J-17-0567_edge RM_HE.ndpi",  # no annotation
    "medical/orig/Scaffan-analysis/PIG-002_J-18-0091_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-002_J-18-0092_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-002_J-18-0094_HE_rescan.ndpi", # bad focus
    "medical/orig/Scaffan-analysis/PIG-003_J-18-0165_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-003_J-18-0166_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-003_J-18-0167_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-003_J-18-0168_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-003_J-18-0169_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-003_J-18-0170_HE.ndpi",
    "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-4_HE_parenchyme.ndpi",
    "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-3 _HE_parenchyme.ndpi",
    "medical/orig/Scaffan-analysis/PIG-004_BBJ-004-2 _HE_parenchyme.ndpi",
    "medical/orig/Scaffan-analysis/PIG-005_J-18-0630_HE_LLL per decell.ndpi",
    "medical/orig/Scaffan-analysis/PIG-005_J-18-0631_HE_LML per decell.ndpi",
    "medical/orig/Scaffan-analysis/PIG-005_J-18-0633_HE_PRML per decell.ndpi",
    "medical/orig/Scaffan-analysis/PIG-008_P008 LL-C_HE_parenchyme centr..ndpi",
    "medical/orig/Scaffan-analysis/PIG-008_P008 LL-P_HE_parenchyme perif..ndpi",
    #
    # "medical/orig/Scaffan-analysis-hů/PIG-001_J-17-0571_LM central_HE.ndpi", # Test diacritics
    #
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-014_Pilsen_HE_LM-P.ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-013_Pilsen_HE_013-LL-P.ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pilsen_Pig-015_HE_RLP_7.ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pilsen_Pig-015_HE_RMP_4.ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-014_Pilsen_HE_RL-C.ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-010_Pilsen_HE_RM-P (1).ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-013_Pilsen_HE_013-LL-C (1).ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-009_Pilsen_HE_009-LM-P.ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-010_Pilsen_HE_LL-P (1).ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-011_Pilsen_HE_LL-P (1).ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-009_Pilsen_HE_009-RL-C (1).ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/PIG-012_Pilsen_HE_LLP (1).ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/Pig-011_Pilsen_HE_RM-P (1).ndpi',
    'medical/orig/Scaffan-analysis-whole-scan-addition/PIG-012_Pilsen_HE_LMC (1).ndpi',
]

for fn in fns:
    if Path(fn).is_absolute():
        fnp = fn
    else:
        fnp = io3d.datasets.join_path(fn, get_root=True)
    set_same(mainapp, fnp)
    mainapp.run_lobuluses(None)
