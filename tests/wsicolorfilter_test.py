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
import sklearn.mixture


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

    chsv_data = []

    # Get pixels from all annotations in one table. Code the output colors into numbers according to `color_ids`

    # id = ids[0]
    for id in ids:
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
        img_chsv = rgb_image_to_chsv(img)
        datai = img_chsv[mask > 0]
        # codes = np.zeros([1,datai.shape[1]])
        # codes[:] = color_ids[color_hexacode]
        codes = np.array([color_ids[color_hexacode]] * datai.shape[0]).reshape([-1, 1])
        datai = np.concatenate((datai, codes), axis=1)

        chsv_data.append(datai)

    chsv_data = np.concatenate(chsv_data, axis=0)

    # Do the training
    models = []
    for hexa, color_id in color_ids.items():
        chsv_data_i = chsv_data[chsv_data[:,-1] == color_id][:,:-1]
        model = sklearn.mixture.BayesianGaussianMixture()

        model.fit(chsv_data_i)
        model


    #


    # Change the color
    id = ids[0]
    views = anim.get_views(annotation_ids=[id], margin=10)
    img = views[0].get_region_image(as_gray=False)
    mask = views[0].get_annotation_region_raster(id)

    img_chsv = rgb_image_to_chsv(img)
    sh = img_chsv.shape
    # flatten the image (roll the x and y axes to the end, squeeze, roll back)
    chsv_data2 = np.moveaxis(np.moveaxis(img_chsv, 2, 0).reshape([sh[2], -1]), 0, 1)
    # chsv_proba = model.predict_proba(chsv_data2)
    chsv_proba = model.score_samples(chsv_data2)

    chsv_proba_img = chsv_proba.reshape(sh[:2])



    plt.imshow(img)
    # plt.imshow(chsv_proba_img)
    plt.contour(mask)
    plt.show()

    # plt.imshow(img)
    plt.imshow(chsv_proba_img)
    plt.contour(mask)
    plt.show()

def rgb_image_to_chsv(img):
    img_hsv = rgb2hsv(img[:, :, :3])
    img_chsv = hue_to_continuous_2d(img_hsv)
    return img_chsv


def hue_to_continuous_2d(img):
    """Takes hsv image and returns ´hhsv´ format, where hue is replaced by 2 values - sin(hue) and cos(hue)"""
    hue = np.expand_dims(img[:, :, 0], -1)
    hue_x = np.cos(hue * 2 * np.pi)
    hue_y = np.sin(hue * 2 * np.pi)
    img = np.concatenate((hue_x, hue_y, img[:, :, 1:]), axis=-1)
    return img
