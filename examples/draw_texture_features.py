# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import pytest
import scaffan
import io3d  # just to get data
import scaffan.image as scim
from typing import List
import exsu
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

# file_path = Path()

import scaffan.slide_segmentation

# def draw_whole_slide_segmentation_get_texture_features():
margin = -0.45
odir = Path(__file__).parent / "slide_texture_features_image/"
print(f"report dir={odir.absolute()}")

fn = io3d.datasets.joinp(
    "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
)
logger.debug(f"input path={fn.absolute()}, exists={fn.exists()}")
fn = str(fn)
logger.debug(f"report dir={odir.absolute()}")
fn = str(fn)

report = exsu.Report(
    outputdir=odir,
    show=False,
    # additional_spreadsheet_fn=odir/"report.xlsx"
    level=10,
)
seg = scaffan.slide_segmentation.ScanSegmentation(report=report)
# seg_method = "GLCMTFS"
# seg_method = "U-Net"
seg_method = "HCTFS"
# seg.parameters.param("Segmentation Method").setValue()
seg.parameters.param("Segmentation Method").setValue(seg_method)
# dir(seg)
anim = scaffan.image.AnnotatedImage(fn)
# seg.init(anim.get_full_view(margin=margin))

# # Get the actual 500x500 um view with default resolution
# v0 = anim.get_full_view(margin=margin)
# v2 = anim.get_view(location = v0.region_location + np.array([2600, 4400]),
#                    # level=seg.level,
#                    size_mm=[0.5, 0.5])
# seg.init(v2)
# v3 = v2.to_level(seg.level)

# Get the Signal paper resolution 10um per pixel size 255
# Semantic Segmentation
v0 = anim.get_full_view(margin=margin)
v3 = anim.get_view(
    location=v0.region_location + np.array([2200, 4400]),
    # level=seg.level,
    pixelsize_mm=[0.01, 0.01],
    # size_on_level=[255, 255],
    size_on_level=[100, 100],
    # size_mm=[0.5, 0.5]
)


feat = seg._get_features(v3)
logger.debug(f"shape={feat.shape}, pixelsize={v3.region_pixelsize}")

import skimage.io

for i in range(0, feat.shape[2]):
    skimage.io.imsave(f"features_{seg_method}_{i:03}.png", feat[:, :, i])

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))
axs = axs.flatten()

suptitles = [
    "Red",
    "Green",
    "Blue",
    "Gaussian 2px",
    "Gaussian 5px",
    "Sobel",
    "Gaussian of Sobel 2px",
    "Gaussian of Sobel 5px",
    "Median of Sobel 10px",
]
for i in range(0, feat.shape[2]):
    axs[i].imshow(feat[:, :, i], cmap="gray")
    axs[i].set_title(suptitles[i])
plt.tight_layout()
plt.savefig("semantic_segmentation_hctf_titles.png", dpi=400)
plt.savefig("semantic_segmentation_hctf_titles.pdf")
plt.show()
# print("asf")
