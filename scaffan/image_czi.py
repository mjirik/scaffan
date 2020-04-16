# /usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

from loguru import logger
import numpy as np
import skimage.transform


def get_py_slices(subb, requested_start, requested_size, output_downscale_factor:int=1):
    """


    return: isconj, slice_subb, slice_requested, real_size_requested

    isconj is bool. It is true if the tile is in requested area
    real_size is unpredictable because if output downcale is 2 the division can be odd.

    """
    s_st = np.asarray(subb.start[-3:-1])
    s_sh = np.asarray(subb.shape[-3:-1])
    r_st = np.asarray(requested_start)
    r_sh = np.asarray(requested_size) * output_downscale_factor
    # r_sh_out = np.asarray(requested_size)
    odf = output_downscale_factor

    # real_start_subb = start_subb - (start_requ + shape_requ)
    # real_end_subb   = start_subb + (shape_subb - start_requ)

    isconj = (r_st + r_sh - s_st > 0).all() and (s_st + s_sh - r_st > 0).all()
    if isconj:

        st_in_s = np.max(np.vstack([r_st - s_st, [0, 0]]), axis=0)
        #         print([r_st - s_st, [0, 0]], f"st_in_s={st_in_s}")
        sp_in_s = np.min(np.vstack([r_st + r_sh - s_st, s_sh]), axis=0)
        st_in_r = np.max(np.vstack([(s_st - r_st)/odf, [0, 0]]), axis=0).astype(int)
        sp_in_r = np.min(np.vstack([(s_st - r_st)/odf + s_sh/odf, r_sh/odf]), axis=0).astype(int)
        if ((s_st - r_st) % odf != [0, 0]).any():
            logger.warning("Problem with downlscale factor. Indices should be int")
        if (r_sh % odf != [0, 0]).any():
            logger.warning("Problem with downlscale factor. Indices should be int")
        sl_s = (
            slice(st_in_s[0], sp_in_s[0]),
            slice(st_in_s[1], sp_in_s[1])
        )
        sl_r = (
            slice(st_in_r[0], sp_in_r[0]),
            slice(st_in_r[1], sp_in_r[1])
        )
        size_r = (
            -(st_in_r[0] - sp_in_r[0]),
            -(st_in_r[1] - sp_in_r[1])
        )

    else:
        sl_s = None
        sl_r = None
        size_r = None
    return isconj, sl_s, sl_r, size_r


def read_region_level0(czi, location, size, downscale_factor=1):
    requested_start = location
    requested_size = size
    output = np.zeros(list(requested_size) + [czi.shape[-1]])
    subbs = []
    for subb in czi.subblocks():
        isconj, sl_s, sl_r, sz_r = get_py_slices(subb, requested_start, requested_size, output_downscale_factor=downscale_factor)

        if isconj:

            subbs.append(subb)

            #             plt.figure()
            #             plt.imshow(subb.data()[0,0,0,:,:,:])
            #             plt.title(f"{subb.start}, {subb.shape}, {subb.stored_shape}")
            # there are several blocks covering the location. Their resolution is the same but the size differes.

            #             print(f"{subb.start}, {subb.shape}, {subb.stored_shape}, [{sl_s[0].start}:{sl_s[0].stop}, {sl_s[1].start}:{sl_s[1].stop}], [{sl_r[0].start}:{sl_r[0].stop}, {sl_r[1].start}:{sl_r[1].stop}]")
            if subb.shape == subb.stored_shape:
                sd = subb.data()
                img = sd[..., sl_s[0], sl_s[1], :]
                logger.debug(img.shape)
                axlist=tuple(range(img.ndim - 3))
                logger.debug(axlist)
                img = np.squeeze(img, axis=axlist)

                img_smaller = skimage.transform.resize(
                    img,
                    output_shape=(sz_r[0], sz_r[1], sd.shape[-1]),
                    preserve_range=True
                ).astype(img.dtype)
                # threr are almost same outputs. The difference is in size of the images
                # there can be 1 pixel error due to integer division
                # img_smaller_alternative = skimage.transform.downscale_local_mean(
                #     img,
                #     factors=(downscale_factor, downscale_factor, 1))
                output[sl_r] = img_smaller
                print(
                    f"{subb.start}, {subb.shape}, {subb.stored_shape}, [{sl_s[0].start}:{sl_s[0].stop}, {sl_s[1].start}:{sl_s[1].stop}], [{sl_r[0].start}:{sl_r[0].stop}, {sl_r[1].start}:{sl_r[1].stop}]")
    #             break
    #         else:
    #             print(f"{subb.start}, {subb.shape}")
    return output
