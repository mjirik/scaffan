# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for texrure analysis.
"""
import logging

logger = logging.getLogger(__name__)

# import warnings
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def tile_centers(image_shape, tile_size):
    tile_size2 = [int(tile_size[0] / 2), int(tile_size[1] / 2)]
    centers = []
    for x0 in range(0, image_shape[0], tile_size[0]):
        for x1 in range(0, image_shape[1], tile_size[1]):
            centers.append([x0 + tile_size2[0], x1 + tile_size2[1]])
    return centers


def tiles_processing(image, fcn, tile_size, fcn_output_n=None, dtype=np.int8):
    shape = list(image.shape)
    if fcn_output_n is not None:
        shape.append(fcn_output_n)
    output = np.zeros(shape, dtype=dtype)
    for x0 in range(0, image.shape[0], tile_size[0]):
        for x1 in range(0, image.shape[1], tile_size[1]):
            sl = (slice(x0, x0 + tile_size[0]), slice(x1, x1 + tile_size[1]))
            img = image[sl]
            output[sl] = fcn(img)

    else:
        return output


def get_feature_and_predict(img, fv_function, classif):
    fv = fv_function(img)
    return classif.predict([fv])[0]


def select_texture_patch_centers_from_one_annotation(
    anim, title, tile_size, level, step=50
):
    if not np.isscalar(tile_size):
        if tile_size[0] == tile_size[1]:
            tile_size = tile_size[0]
        else:
            # it would be possible to add factor (1./tile_size) into distance transform
            raise ValueError(
                "Both sides of tile should be the same. Other option is not implemented."
            )
    annotation_ids = anim.select_annotations_by_title(title)
    view = anim.get_views(annotation_ids, level=level)[0]
    mask = view.get_annotation_region_raster(title)

    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", "low contrast image")
    dst = scipy.ndimage.morphology.distance_transform_edt(mask)
    middle_pixels = dst > (tile_size / 2)
    # x_nz, y_nz = nonzero_with_step(middle_pixels, step)
    y_nz, x_nz = nonzero_with_step(middle_pixels, step)
    nz_global_px = view.coords_view_px_to_glob_px(x_nz, y_nz)
    # anim.
    return nz_global_px


def nonzero_with_step(data, step):
    # print(data.shape)
    datastep = data[::step, ::step]
    # print(datastep.shape)
    nzx, nzy = np.nonzero(datastep)

    return nzx * step, nzy * step


class TextureSegmentation:
    def __init__(self, feature_function=None, classifier=None):
        self.tile_size = [256, 256]
        self.tile_size1 = 256
        self.level = 1
        self.step = 64
        self.data = []
        self.target = []
        if feature_function is None:
            import scaffan.texture_lbp as salbp

            feature_function = salbp.lbp_fv
        self.feature_function = feature_function
        if classifier is None:
            import scaffan.texture_lbp as salbp

            classifier = salbp.KLDClassifier()
        self.classifier = classifier

        # n_points = 8
        # radius = 3
        # METHOD = "uniform"
        # self.feature_function_args = [n_points, radius, METHOD]
        pass

    def set_tile_size(self, tile_size1):
        self.tile_size = [tile_size1, tile_size1]
        self.tile_size1 = tile_size1

    def get_tile_centers(self, anim, annotation_id, return_xy=False):
        """
        Calculate centers for specific annotation.
        :param anim:
        :param annotation_id:
        :return: [[x0, y0], [x1, y1], ...]
        """
        patch_centers1 = select_texture_patch_centers_from_one_annotation(
            anim,
            annotation_id,
            tile_size=self.tile_size1,
            level=self.level,
            step=self.step,
        )
        if return_xy:
            return patch_centers1
        else:
            patch_centers1_points = list(zip(*patch_centers1))
            return patch_centers1_points

    def get_patch_view(self, anim, patch_center=None, annotation_id=None):
        if patch_center is not None:
            view = anim.get_view(
                center=[patch_center[0], patch_center[1]],
                level=self.level,
                size=self.tile_size,
            )
        elif patch_center is not None:
            annotation_ids = anim.select_annotations_by_title(
                title=annotation_id, level=self.level, size=self.tile_size
            )
            view = anim.get_views(annotation_ids)[0]

        return view

    def show_tiles(self, anim, annotation_id, tile_ids):
        """
        Show tiles from annotation selected by list of its id
        :param anim:
        :param annotation_id:
        :param tile_ids: list of int, [0, 5] means show first and sixth tile
        :return:
        """
        patch_center_points = self.get_tile_centers(anim, annotation_id)
        for id in tile_ids:
            view = self.get_patch_view(anim, patch_center_points[id])
            plt.figure()
            plt.imshow(view.get_region_image(as_gray=True), cmap="gray")

    def add_training_data(self, anim, annotation_id, numeric_label, show=False):
        patch_center_points = self.get_tile_centers(anim, annotation_id)

        for patch_center in patch_center_points:
            view = self.get_patch_view(anim, patch_center)
            imgray = view.get_region_image(as_gray=True)
            self.data.append(self.feature_function(imgray))
            self.target.append(numeric_label)

        if show:
            annotation_ids = anim.select_annotations_by_title(title=annotation_id)
            view = anim.get_views(annotation_ids)[0]
            view.imshow()
            lst = list(zip(*patch_center_points))
            x, y = lst
            view.plot_points(x, y)
        return patch_center_points

    def fit(self):
        self.classifier.fit(self.data, self.target)
        pass

    def predict(self, view, show=False):
        test_image = view.get_region_image(as_gray=True)

        tile_fcn = lambda img: get_feature_and_predict(
            img, self.feature_function, self.classifier
        )
        seg = tiles_processing(test_image, tile_fcn, tile_size=self.tile_size)

        if show:
            centers = tile_centers(test_image.shape, tile_size=self.tile_size)
            import skimage.color

            plt.imshow(skimage.color.label2rgb(seg, test_image))
            x, y = list(zip(*centers))
            plt.plot(x, y, "xy")
            # view.plot_points()
        return seg
