import io3d
from loguru import logger
import pytest
import scaffan.image as scim
from scaffan import image_czi
scim.import_openslide()
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
import scaffan.image_intensity_rescale
import re
import sklearn.mixture
import sklearn.neighbors


def test_change_color_of_wsi():
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
        # model = sklearn.neighbors.KernelDensity()

        model.fit(chsv_data_i)
        # model


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
    chsv_proba2 = model.score_samples(chsv_data2)
    #
    # chsv_proba_img = chsv_proba.reshape(sh[:2])
    chsv_proba2_img = chsv_proba2.reshape(sh[:2])
    slope = 1.


    #    limit under zero and suqeeze data
    #    log(exp(log(score)) + 1)

    chsv_proba2_img_exp = np.exp(chsv_proba2_img)
    # chsv_proba2_img = scaffan.image_intensity_rescale.sigmoidal(np.log(chsv_proba2_img_exp + 1))
    # chsv_proba2_img = 1 - (np.exp(-np.log(chsv_proba2_img_exp + 1) )) ## this seems to be ok
    chsv_proba2_img_log = np.log(chsv_proba2_img_exp + 1)
    # chsv_proba2_img = 1 / (1 + np.exp(-(chsv_proba2_img_exp)))
    chsv_proba2_img = 1-np.exp(-chsv_proba2_img_log * slope) # best
    # chsv_proba2_img = 1-np.exp(-chsv_proba2_img* slope)
    # chsv_proba2_img = chsv_proba2_img_exp* slope
    # chsv_proba2_img[chsv_proba2_img > 1.] = 1

    #

    new_img = change_color_using_probability(img, chsv_proba2_img, "#ffffff")
    x = np.linspace(-5,5)
    y = scaffan.image_intensity_rescale.sigmoidal(x* slope)
    # plt.plot(x,y)
    # plt.show()
    #
    # plt.hist(chsv_proba2.flatten(), bins=np.linspace(0, 1, 20))
    # plt.show()
    plti = 420

    if False:
        plt.subplot(plti + 1)
        plt.imshow(img)
        plt.contour(mask)

        plt.subplot(plti + 2)
        plt.imshow(new_img)
        plt.contour(mask)

        plt.subplot(plti + 3)
        plt.hist(chsv_proba2.flatten(), bins=np.linspace(-10,4,20))

        plt.subplot(plti + 4)
        plt.imshow(chsv_proba2.reshape(sh[:2]))
        plt.contour(mask)
        plt.colorbar()

        plt.subplot(plti + 5)
        plt.hist(chsv_proba2_img.flatten(), bins=np.linspace(-2,2,30))
        # plt.colorbar()

        plt.subplot(plti+ 6)
        # plt.imshow(img)
        plt.imshow(chsv_proba2_img)
        plt.colorbar()

        plt.subplot(plti + 7)
        plt.hist(chsv_proba2_img_exp.flatten(), bins=np.linspace(-2,2,30))

        plt.subplot(plti+ 8)
        plt.imshow(chsv_proba2_img_exp)
        plt.colorbar()
    # plt.contour(mask)



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


def change_color_using_probability(img_rgb, img_proba, target_color):
    import matplotlib.colors
    # target_color = '#B4FBB8'
    color_rgb = np.asarray(matplotlib.colors.to_rgb(target_color))
    color_hsv = rgb2hsv(color_rgb.reshape([1,1,3]))
    img_hsv = rgb2hsv(img_rgb)
    diff = color_hsv - img_hsv

    # img_hsv - img_hsv - color_hsv
    if img_proba.ndim == 2:
        img_proba = np.expand_dims(img_proba, 2)
    elif img_proba.ndim == 3:
        pass
    else:
        raise ValueError("probability is expected to be 2D or 3D")

    img_proba3 = np.concatenate([img_proba] * 3, axis=2)

    new_img = hsv2rgb(img_hsv + img_proba3 * diff)
    return new_img


def test_change_color():
    sz = [10, 15, 3]
    img_rgb = np.random.random(sz) * 30
    img_rgb[:5,:5,1] += 100
    img_rgb[:5,5:,2] += 100
    img_rgb[5:,5:,2] += 100
    img_rgb[img_rgb>255] = 255
    img_rgb = img_rgb.astype(np.uint8)
    img_proba = np.zeros(sz[:2])
    img_proba[3:8, 3:8] = .5
    img_proba[5:7, 5:7] = 1.

    new_img = change_color_using_probability(img_rgb, img_proba, "#ffff00")
    assert all(np.isclose(new_img[6, 6], [1, 1, 0]))
    # plt.imshow(img_rgb)
    # plt.show()
    # plt.imshow(new_img)
    # plt.show()






def test_log():
    logx = np.linspace(-1000,1000)
    y = 1 / (1 + np.exp(-logx))
    # plt.plot(logx, y)
    # plt.show()


def test_log2():
    x = np.linspace(0,1000, 10000)
    y = 1-np.exp(-np.log(x + 1))
    y2 = 1-np.exp(-x)
    # plt.plot(x, y)
    # plt.plot(x,y2)
    # plt.show()
