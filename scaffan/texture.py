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


def texture_segmentation(image, decision_function, models, tile_size):
    output = np.asarray(image, dtype=np.int8)
    for x0 in range(0, image.shape[0], tile_size[0]):
        for x1 in range(0, image.shape[1], tile_size[1]):
            sl = [
                slice(x0, x0 + tile_size[0]),
                slice(x1, x1 + tile_size[1])
            ]
            output[sl] = decision_function(models, image[sl])

    return output


def select_texture_patch_centers_from_one_annotation(anim, i, tile_size, level, step=50):
    if not np.isscalar(tile_size):
        if tile_size[0] == tile_size[1]:
            tile_size = tile_size[0]
        else:
            # it would be possible to add factor (1./tile_size) into distance transform
            raise ValueError("Both sides of tile should be the same. Other option is not implemented.")
    view = anim.get_view_on_annotation(i, level=level)
    mask = view.get_annotation_region_raster(i)

    dst = scipy.ndimage.morphology.distance_transform_edt(mask)
    middle_pixels = dst > (tile_size / 2)
    nz = nonzero_with_step(middle_pixels, step)
    nz_global_px = view.coords_view_px_to_glob_px(nz)
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

    def get_tile_centers(self, anim, tile):
        patch_centers1 = select_texture_patch_centers_from_one_annotation(anim, tile, tile_size=self.tile_size1,
                                                                          level=self.level, step=self.step)
        patch_centers1_points = list(zip(*patch_centers1))
        return patch_centers1_points

    def get_patch_view(self,anim, patch_center):
        view = anim.get_view(center=[patch_center[0], patch_center[1]], level=self.level, size=self.tile_size)

        return view

    def show_tiles(self, anim, tile, tile_ids):
        patch_center_points = self.get_tile_centers(anim, tile)
        for id in tile_ids:
            view = self.get_patch_view(anim, patch_center_points[id])
            plt.figure()
            plt.imshow(view.get_region_image(as_gray=True), cmap="gray")

    def add_training_data(self, anim, tile, numeric_label):
        patch_center_points = self.get_tile_centers(anim, tile)

        for patch_center in patch_center_points:
            view = self.get_patch_view(anim, patch_center)
            imgray = view.get_region_image(as_gray=True)
            self.refs.append([numeric_label, self.feature_function(imgray, *self.feature_function_args)])


    def fit(self, view, show=False):
        test_image = view.get_region_image(as_gray=True)
        import scaffan.texture_lbp as salbp
        seg = texture_segmentation(test_image, salbp.match, self.refs, tile_size=self.tile_size)
        if show:
            import skimage.color
            plt.imshow(skimage.color.label2rgb(seg, test_image))
        return seg

