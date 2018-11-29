# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for texrure analysis.
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage


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
