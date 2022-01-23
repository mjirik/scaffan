import io3d
from loguru import logger
import pytest
import scaffan.image as scim
import scaffan.algorithm
from scaffan.wsi_color_filter import WsiColorFilter, change_color_using_probability
from scaffan import image_czi
scim.import_openslide()
from scaffan import image
import matplotlib.pyplot as plt
import numpy as np


def test_change_color_whole_scaffan():
    fn = io3d.datasets.join_path(
        "medical/orig/scaffan-analysis-czi/J7_5/J7_5_b_test.czi", get_root=True
    )
    mainapp = scaffan.algorithm.Scaffan()
    mainapp.set_input_file(fn)
    # mainapp.set_annotation_color_selection(
    #     "#FFFF00", override_automatic_lobulus_selection=True
    # )
    mainapp.init_run()
    # mainapp.color_filter.set_anim_params(mainapp.anim)
    view, img = mainapp.get_preview()
    plt.imshow(img)
    plt.show()

    # mainapp.set_parameter("Processing;Skeleton Analysis", False)
    # mainapp.set_parameter("Processing;Texture Analysis", False)


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
    color_filter = WsiColorFilter()
    color_filter.init_color_filter_by_anim(anim)
    color_filter.slope = 10000
    color_filter.proportion = 1.

    size_px = 100
    size_on_level = [size_px, size_px]

        # model
    color_filter.init_color_filter_by_anim(anim)

    #

    regex = " *convert *color *to *(#[a-fA-F0-9]{6})"
    ids = anim.select_annotations_by_title_regex(regex)

    # Change the color
    id = ids[0]
    views = anim.get_views(annotation_ids=[id], margin=10)
    img = views[0].get_region_image(as_gray=False)
    mask = views[0].get_annotation_region_raster(id)
    new_img, probas = color_filter.img_processing(img, return_proba=True)


    ####
    # x = np.linspace(-5,5)
    # y = scaffan.image_intensity_rescale.sigmoidal(x* slope)
    # plt.plot(x,y)
    # plt.show()
    #
    # plt.hist(chsv_proba2.flatten(), bins=np.linspace(0, 1, 20))
    # plt.show()

    # if True:
    assert np.max(list(probas.values())[0]) > 0.6
    assert np.min(list(probas.values())[0]) < 0.4

    if True:
        plti = 430
        plt.subplot(plti + 1)
        plt.imshow(img)
        plt.contour(mask)
        sh = img.shape

        plt.subplot(plti + 2)
        plt.imshow(new_img)
        plt.title("new_img")
        plt.contour(mask)

        key = list(probas.keys())[0]
        chsv_proba2_img = probas[key]

        plt.subplot(plti + 3)
        plt.imshow(np.exp(chsv_proba2_img))
        plt.colorbar()
        # plt.hist(chsv_proba2.flatten(), bins=np.linspace(-10,4,20))
        #
        plt.subplot(plti + 4)
        plt.hist(np.exp(chsv_proba2_img).flatten(), histtype = 'step',
                   cumulative = True, density = True
                 )
        plt.hist(np.exp(chsv_proba2_img).flatten(), histtype = 'step',
                 cumulative = False, density = True
                 )
        # plt.imshow(chsv_proba2.reshape(sh[:2]))
        # plt.contour(mask)
        # plt.colorbar()

        plt.subplot(plti + 5)
        plt.hist(chsv_proba2_img.flatten(), bins=np.linspace(-2, 2, 30))
        # plt.colorbar()

        plt.subplot(plti + 6)
        # plt.imshow(img)
        plt.imshow(chsv_proba2_img)
        plt.title("original proba")
        plt.colorbar()

        plt.subplot(plti + 7)
        mx = np.max(np.exp(chsv_proba2_img))
        print(mx)
        probas_linspace = np.linspace(0, mx, 10000)
        probas_linspace_weighted = color_filter._soft_maximum_limit(probas_linspace)
        plt.plot(probas_linspace, probas_linspace_weighted)


        plt.subplot(plti + 8)
        plt.imshow(color_filter._weighted_proba(chsv_proba2_img))
        plt.colorbar()
        # plt.hist(chsv_proba2_img_exp.flatten(), bins=np.linspace(-2,2,30))
        #
        # plt.subplot(plti+ 8)
        # plt.imshow(chsv_proba2_img_exp)
        # plt.colorbar()
        # plt.contour(mask)
        plt.show()


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
