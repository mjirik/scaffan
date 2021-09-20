import io3d
from loguru import logger
import pytest
import scaffan.image as scim
from scaffan import image_czi
scim.import_openslide()
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv
import re


def test_read_annotation():
    fn = io3d.datasets.join_path(
        "medical/orig/scaffan-analysis-czi/J7_5/J7_5_b_test.czi", get_root=True
    )
    # fn = io3d.datasets.join_path(
    #     "medical/orig/scaffan-analysis-czi/Zeiss-scans/01_2019_11_12__RecognizedCode.czi",
    #     get_root=True)
    logger.debug("filename {}".format(fn))
    anim = scim.AnnotatedImage(fn)
    logger.debug(anim.annotations)
    size_px = 100
    size_on_level = [size_px, size_px]
    logger.debug(anim.id_by_titles)
    regex = " *convert *color *to *#([a-fA-F0-9]{6})"
    ids = anim.select_annotations_by_title_regex(regex)
    logger.debug(ids)

    color_hexacodes = [
        re.findall(regex, anim.annotations[id]["title"])[0] for id in ids
    ]

    color_ids = {value:number for number, value in enumerate(np.unique(
        color_hexacodes
    ))
    }

    id = ids[0]
    views = anim.get_views(annotation_ids=[id])
    mask = views[0].get_annotation_region_raster(id)
    color_hexacode = re.findall(regex, anim.annotations[id]["title"])[0]



    from skimage.color import rgb2hsv
    # view = anim.get_view(
    #     location_mm=[10, 11],
    #     size_on_level=size_on_level,
    #     level=5,
    #     # size_mm=[0.1, 0.1]
    # )
    img = views[0].get_region_image(as_gray=False)
    img_hsv = rgb2hsv(img[:, :, :3])
    img_chsv = hue_to_continuous_2d(img_hsv)
    plt.imshow(img)
    plt.contour(mask)
    plt.show()


def hue_to_continuous_2d(img):
    """Takes hsv image and returns ´hhsv´ format, where hue is replaced by 2 values - sin(hue) and cos(hue)"""
    hue = np.expand_dims(img[:, :, 0], -1)
    hue_x = np.cos(hue * 2 * np.pi)
    hue_y = np.sin(hue * 2 * np.pi)
    img = np.concatenate((hue_x, hue_y, img[:, :, 1:]), axis=-1)
    return img
