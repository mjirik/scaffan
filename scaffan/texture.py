# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for texrure analysis.
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def texture_segmentation(image, decision_function, models, tile_size, return_centers=False):
    if return_centers:
        tile_size2 = [int(tile_size[0] / 2), int(tile_size[1] / 2)]
        centers = []
    output = np.asarray(image, dtype=np.int8)
    for x0 in range(0, image.shape[0], tile_size[0]):
        for x1 in range(0, image.shape[1], tile_size[1]):
            sl = [
                slice(x0, x0 + tile_size[0]),
                slice(x1, x1 + tile_size[1])
            ]
            output[sl] = decision_function(models, image[sl])
            if return_centers:
                centers.append([x0 + tile_size2[0], x1 + tile_size2[1]])

    if return_centers:
        return output, centers
    else:
        return output


def select_texture_patch_centers_from_one_annotation(anim, title, tile_size, level, step=50):
    if not np.isscalar(tile_size):
        if tile_size[0] == tile_size[1]:
            tile_size = tile_size[0]
        else:
            # it would be possible to add factor (1./tile_size) into distance transform
            raise ValueError("Both sides of tile should be the same. Other option is not implemented.")
    annotation_ids = anim.select_annotations_by_title(title)
    view = anim.get_views(annotation_ids, level=level)[0]
    mask = view.get_annotation_region_raster(title)

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
    def __init__(self):
        self.tile_size = [256, 256]
        self.tile_size1 = 256
        self.level = 1
        self.step = 64
        self.refs = []
        import scaffan.texture_lbp as salbp
        self.feature_function = salbp.local_binary_pattern

        n_points = 8
        radius = 3
        METHOD = "uniform"
        self.feature_function_args = [n_points, radius, METHOD]
        pass

    def get_tile_centers(self, anim, annotation_id, return_xy=False):
        """
        Calculate centers for specific annotation.
        :param anim:
        :param annotation_id:
        :return: [[x0, y0], [x1, y1], ...]
        """
        patch_centers1 = select_texture_patch_centers_from_one_annotation(anim, annotation_id, tile_size=self.tile_size1,
                                                                          level=self.level, step=self.step)
        if return_xy:
            return patch_centers1
        else:
            patch_centers1_points = list(zip(*patch_centers1))
            return patch_centers1_points

    def get_patch_view(self, anim, patch_center=None, annotation_id=None):
        if patch_center is not None:
            view = anim.get_view(center=[patch_center[0], patch_center[1]], level=self.level, size=self.tile_size)
        elif patch_center is not None:
            annotation_ids = anim.select_annotations_by_title(title=annotation_id, level= self.level, size=self.tile_size)
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
            self.refs.append([numeric_label, self.feature_function(imgray, *self.feature_function_args)])

        if show:
            annotation_ids = anim.select_annotations_by_title(title=annotation_id)
            view = anim.get_views(annotation_ids)[0]
            view.imshow()
            lst = list(zip(*patch_center_points))
            x, y = lst
            view.plot_points(x, y)
        return patch_center_points


    def fit(self, view, show=False):
        test_image = view.get_region_image(as_gray=True)
        import scaffan.texture_lbp as salbp

        out = texture_segmentation(test_image, salbp.match, self.refs, tile_size=self.tile_size, return_centers=show)

        if show:
            seg, centers = out
            import skimage.color
            plt.imshow(skimage.color.label2rgb(seg, test_image))
            x, y = list(zip(*centers))
            plt.plot(x, y, "xy")
            # view.plot_points()
        else:
            seg = out
        return seg

