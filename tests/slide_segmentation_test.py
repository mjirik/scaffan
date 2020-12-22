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


def test_slide_segmentation_hamamatsu():
    odir = Path(__file__).parent / "slide_seg_SCP003_test_output/"
    print(f"report dir={odir.absolute()}")

    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    run_slide_seg(odir, Path(fn), margin=-0.35, add_biggest=True)
    # assert False


def test_slide_segmentation_zeiss():
    odir = Path(__file__).parent / "slide_seg_Recog_test_output/"
    fn = io3d.datasets.join_path(
        # "medical/orig/scaffan-analysis-czi/Zeiss-scans/01_2019_11_12__RecognizedCode.czi",
        "medical/orig/scaffan-analysis-czi/Zeiss-scans/05_2019_11_12__-1-2.czi",
        get_root=True,
    )
    # seg = run_slide_seg(odir, Path(fn), margin=-0.4)
    seg = run_slide_seg(odir, Path(fn), margin=-0.4)


def run_slide_seg(
    odir: Path, fn: Path, margin: float, check_black_ids=False, add_biggest=False
):
    logger.debug(f"report dir={odir.absolute()}")
    fn = str(fn)

    report = exsu.Report(
        outputdir=odir,
        show=False,
        # additional_spreadsheet_fn=odir/"report.xlsx"
        level=10,
    )
    seg = scaffan.slide_segmentation.ScanSegmentation(report=report)
    # dir(seg)
    anim = scaffan.image.AnnotatedImage(fn)

    seg.init(anim.get_full_view(margin=margin))
    # seg.parameters.param("Segmentation Method").setValue("HCTFS")
    seg.parameters.param("Segmentation Method").setValue("U-Net")
    seg.parameters.param("Save Training Labels").setValue(True)
    if check_black_ids:
        ann_ids_black = seg.anim.get_annotations_by_color("#000000")
        assert 10 in ann_ids_black
        assert 11 in ann_ids_black
    seg.run()
    assert type(seg.full_output_image) == np.ndarray
    assert type(seg.full_raster_image) == np.ndarray
    assert type(seg.whole_slide_training_labels) == np.ndarray
    assert seg.full_raster_image.shape[:2] == seg.full_output_image.shape[:2]
    assert seg.whole_slide_training_labels.shape[:2] == seg.full_output_image.shape[:2]
    if seg.accuracy:
        assert seg.accuracy > 0.2
    if add_biggest:
        seg.add_biggest_to_annotations()
    seg.report.finish_actual_row()
    seg.report.dump()
    return seg
    # plt.imshow(seg.full_prefilter_image)
    # plt.show()
    # plt.imshow(seg.full_output_image)
    # plt.show()
    # plt.imshow(seg.full_raster_image)
    # plt.contour(seg.whole_slide_training_labels)
    # plt.show()
    # plt.imshow(seg.whole_slide_training_labels)
    # plt.show()


def test_get_biggest_lobuli():
    odir = Path(__file__).parent / "slide_seg_SCP003_test_output/"
    print(f"report dir={odir.absolute()}")

    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    logger.debug(f"report dir={odir.absolute()}")
    fn = str(fn)

    report = exsu.Report(
        outputdir=odir,
        show=False
        # additional_spreadsheet_fn=odir/"report.xlsx"
    )
    seg = scaffan.slide_segmentation.ScanSegmentation(report=report)
    # dir(seg)
    anim = scaffan.image.AnnotatedImage(fn)
    seg.init(anim.get_full_view(margin=-0.4))

def test_postprocessing():
    odir = Path(__file__).parent / "slide_seg_SCP003_test_output/"
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    logger.debug(f"report dir={odir.absolute()}")
    fn = str(fn)

    report = exsu.Report(
        outputdir=odir,
        show=False
        # additional_spreadsheet_fn=odir/"report.xlsx"
    )
    seg = scaffan.slide_segmentation.ScanSegmentation(report=report)
    # dir(seg)
    seg.parameters.param("Segmentation Method").setValue("GLCMTFS")
    seg.parameters.param("Save Training Labels").setValue(True)
    im = np.random.random([600, 400])
    sg = (im > 0.3).astype(np.uint8)
    sg[300:320, 100:105] = 2
    imf = seg._labeling_filtration(sg)
    assert imf[0,-1] == 0
    assert imf[300,110] == 1
    # plt.imshow(sg)
    # plt.show()
    # plt.imshow(imf)
    # plt.show()


def test_and_draw_whole_slide_segmentation_get_texture_features():
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
    v3 = anim.get_view(location = v0.region_location + np.array([2200, 4400]),
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
        skimage.io.imsave(f"features_{seg_method}_{i:03}.png", feat[:,:,i])

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8,8))
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
    # plt.show()
    # print("asf")
