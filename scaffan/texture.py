# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for texrure analysis.
"""

from loguru import logger

# import warnings
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import copy
from typing import List
import os.path as op
from pyqtgraph.parametertree import Parameter, ParameterTree
from . import image
from exsu.report import Report
import imma.image
from . import sni_prediction


def tile_centers(image_shape, tile_spacing):
    tile_size2 = [int(tile_spacing[0] / 2), int(tile_spacing[1] / 2)]
    centers = []
    for x0 in range(0, image_shape[0], tile_spacing[0]):
        for x1 in range(0, image_shape[1], tile_spacing[1]):
            centers.append([x0 + tile_size2[0], x1 + tile_size2[1]])
    return centers


def tiles_processing(
    image, fcn, tile_spacing, fcn_output_n=None, dtype=np.int8, tile_size=None
):
    """
    Process image tile by tile. Last tile in every row and avery column may be smaller if modulo of shape of image and
    shape of tile is different from zero. On the border of image tile is smaller according to the edge of the image.

    :param image: input image
    :param fcn: Function used on each tile. Input of this function is just tile image.
    :param tile_spacing: size of tile in pixels
    :param fcn_output_n: dimension of output of fcn()
    :param dtype: output data type
    :param tile_size: [int, int]: size of tile in pixels. Tile size can be set to obtain overlap between two
    neighbour tiles.
    :return:
    """
    # TODO rename inputs

    shape = list(image.shape)
    if fcn_output_n is not None:
        shape.append(fcn_output_n)
    if tile_size is None:
        tile_margin = [0, 0]
    else:
        tile_margin = ((np.asarray(tile_size) - tile_spacing) / 2).astype(np.int)

    output = np.zeros(shape, dtype=dtype)
    for x0 in range(0, image.shape[0], tile_spacing[0]):
        for x1 in range(0, image.shape[1], tile_spacing[1]):
            sl_in = (
                slice(
                    max(x0 - tile_margin[0], 0), x0 + tile_spacing[0] + tile_margin[0]
                ),
                slice(
                    max(x1 - tile_margin[1], 0), x1 + tile_spacing[1] + tile_margin[1]
                ),
            )
            sl_out = (slice(x0, x0 + tile_spacing[0]), slice(x1, x1 + tile_spacing[1]))
            img = image[sl_in]
            output[sl_out] = fcn(img)

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
    mask = view.get_annotation_raster(title)

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


class GLCMTextureMeasurement:
    def __init__(
        self,
        filename_label="",
        pname="Texture Analysis",
        ptype="bool",
        pvalue=True,
        ptip="Run Gray Level Co-occurrence texture analysis after lobulus segmentation is performed",
        texture_label=None,
        working_resolution_mm=0.0000004,
        tile_size=64,
        tile_spacing=32,
        add_cols_to_report: bool = True,
        report_severity_offset=0,
        glcm_levels=64,
        report: Report = None,
        sni_predictor=None,
    ):
        """
        Make texture description.
        It is used twice in scaffan.

        :param filename_label:
        :param pname:
        :param ptype:
        :param pvalue:
        :param ptip:
        :param texture_label: Used to identify where the texture processing is used (central vein, whole_slide, ...)
        :param working_resolution_mm:
        :param tile_size:
        :param tile_spacing:
        """
        self.sni_predictor = sni_predictor
        params = [
            {
                "name": "Tile Size",
                "type": "int",
                # "value": 64,
                "value": tile_size,
                "suffix": "px",
            },
            {
                "name": "Tile Spacing",
                "type": "int",
                # "value": 32,
                "value": tile_spacing,
                "suffix": "px",
            },
            {
                "name": "Working Resolution",
                "type": "float",
                # "value": 0.000001,
                # "value": 0.0000005,
                # "value": 0.0000004,
                "value": working_resolution_mm,
                "suffix": "m",
                "siPrefix": True,
            },
            {"name": "GLCM Levels", "type": "int", "value": glcm_levels},
        ]
        if self.sni_predictor is not None:
            params.append(self.sni_predictor.parameters)
        self.texture_label = texture_label

        self.parameters = Parameter.create(
            name=pname,
            type=ptype,
            value=pvalue,
            tip=ptip,
            children=params,
            expanded=False,
        )
        self.report: Report = report
        self.filename_label = filename_label
        self.add_cols_to_report: bool = add_cols_to_report
        self.annotation_id = None
        self.lobulus_segmentation = None
        self.report_severity_offset = report_severity_offset

    def set_report(self, report: Report):
        self.report = report

    def set_input_data(
        self, view: image.View, annotation_id: int = None, lobulus_segmentation=None
    ):
        self.anim = view.anim
        self.annotation_id = annotation_id
        self.parent_view = view
        logger.trace(f"lobulus segmentation {lobulus_segmentation}")
        self.lobulus_segmentation = lobulus_segmentation

    def run(self, recalculate_view=True):
        """

        :param recalculate_view: if False do not calculate other view paramters and use this one
        :return:
        """

        # title = "test3"
        # title = "test2"
        # title = "test1"
        # views = self.anim.get_views_by_title(self.annotation_id, level=0)
        pxsize_mm = [self.parameters.param("Working Resolution").value() * 1000] * 2
        tile_size = [self.parameters.param("Tile Size").value()] * 2
        tile_spacing = [self.parameters.param("Tile Spacing").value()] * 2
        levels = self.parameters.param("GLCM Levels").value()
        # view = views[0].to_pixelsize(pxsize_mm)
        if self.texture_label is not None:
            texture_label_fn = "_" + self.texture_label
        else:
            texture_label_fn = ""
        if recalculate_view:
            view = self.parent_view.to_pixelsize(pxsize_mm)
        else:
            view = self.parent_view

        texture_image = view.get_region_image(as_gray=True)

        fn_ann_id = "" if self.annotation_id is None else f"_{self.annotation_id}"
        fn_id = "{}{}{}".format(self.filename_label, fn_ann_id, texture_label_fn)
        if self.report is not None:
            self.report.imsave(
                "texture_input_image_{}.png".format(fn_id),
                (texture_image * 255).astype(np.uint8),
                level=60 + self.report_severity_offset,
                level_npz=40 + self.report_severity_offset,
                level_skimage=20 + self.report_severity_offset,
            )
        energy = tiles_processing(
            1 - texture_image,
            fcn=lambda img: texture_glcm_features(img, levels),
            tile_spacing=tile_spacing,
            fcn_output_n=3,
            dtype=None,
            tile_size=tile_size,
        )
        # seg = texseg.predict(views[0], show=False, function=texture_energy)
        fig = plt.figure(figsize=(10, 12))
        plt.subplot(221)
        # grayscale image is there because of travis memory error
        img = view.get_region_image(as_gray=True)
        plt.imshow(img)
        # if self.annotation_id is not None:
        view.plot_annotations(self.annotation_id)
        if self.lobulus_segmentation is not None:
            seg = (
                imma.image.resize_to_shape(
                    self.lobulus_segmentation, shape=img.shape[:2], order=0
                )
                == 1
            )
            plt.contour(seg)
            plt.title("original image")
        plt.subplot(222)
        plt.title("GLCM energy")
        image.imshow_with_colorbar(energy[:, :, 0])
        plt.subplot(223)
        plt.title("GLCM homogeneity")
        image.imshow_with_colorbar(energy[:, :, 1])
        plt.subplot(224)
        plt.title("GLCM correlation")
        image.imshow_with_colorbar(energy[:, :, 2])
        mx = np.max(energy, axis=(0, 1))
        # mn = np.min(energy, axis=(0, 1))
        logger.debug(f"GLCM max energy, homogeneity, correlation: {mx}")
        # plt.colorbar()
        if self.report is not None:
            self.report.savefig_and_show(
                "glcm_features_{}.png".format(fn_id),
                fig,
                level=60 + self.report_severity_offset,
            )
        # if close_figs:
        #     plt.close(fig)
        # plt.savefig("glcm_features_{}.png".format(title))

        fig = plt.figure()
        plt.imshow(energy)
        if self.report is not None:
            self.report.savefig_and_show(
                "glcm_features_color_{}.png".format(fn_id),
                fig,
                level=60 + self.report_severity_offset,
            )
        # if close_figs:
        #     plt.close(fig)

        e0 = energy[:, :, 0]
        e1 = energy[:, :, 1]
        e2 = energy[:, :, 2]
        self.measured_features = energy

        if self.lobulus_segmentation is None:
            logger.debug(f"No lobulus segmentation given")
            seg = (slice(None), slice(None))
        if self.texture_label is None:
            texture_label_stats = ""
        else:
            texture_label_stats = f" {self.texture_label}"

        row = {}
        row = make_stats(f"GLCM Energy{texture_label_stats}", e0[seg], row)
        row = make_stats(f"GLCM Homogenity{texture_label_stats}", e1[seg], row)
        row = make_stats(f"GLCM Correlation{texture_label_stats}", e2[seg], row)
        # row = {
        #     "GLCM Energy": np.mean(e0[seg]),
        #     "GLCM Homogenity": np.mean(e1[seg]),
        #     "GLCM Correlation": np.mean(e2[seg]),
        # }
        if self.sni_predictor is not None:
            # texture processing is called twice
            self.sni_predictor.predict_area(row)
        if self.report is not None and self.add_cols_to_report:
            self.report.add_cols_to_actual_row(row)
        logger.debug(
            f"GLCM textures for id {self.annotation_id if self.annotation_id is not None else '-'} finished"
        )

        # plt.show()


def make_stats(prefix: str, data, dct=None):
    """
    Calculate mean, variance and 10, 25, 75 and 90 percentiles. Output
    is stored to dict with key starting with `prefix`
    :param prefix: prefix of output dict key
    :param data: numeric data to analyze
    :param dct: input dict. Empty dict is created if None is given.
    :return:
    """
    if dct is None:
        dct = {}

    dct[prefix] = np.mean(data)
    dct[f"{prefix} var"] = np.var(data)
    dct[f"{prefix} p10"] = np.percentile(data, 10)
    dct[f"{prefix} p25"] = np.percentile(data, 25)
    dct[f"{prefix} p50"] = np.percentile(data, 50)
    dct[f"{prefix} p75"] = np.percentile(data, 75)
    dct[f"{prefix} p90"] = np.percentile(data, 90)
    return dct


class TextureSegmentation:
    def __init__(self, feature_function=None, classifier=None):

        params = [
            {"name": "Tile Size", "type": "int", "value": 256},
            # {
            #     "name": "Working Resolution",
            #     "type": "float",
            #     "value": 0.001,
            #     "suffix": "m",
            #     "siPrefix": True
            #
            # }
        ]

        self.parameters = Parameter.create(
            name="Texture Processing", type="group", children=params
        )
        self.parameters
        self.tile_size = None
        self.tile_size1 = None
        self.set_tile_size(self.parameters.param("Tile Size").value())
        self.parameters.param("Tile Size").sigValueChanged.connect(
            self._seg_tile_size_params
        )

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
        logger.debug("texture run finished")

    def _seg_tile_size_params(self):
        self.set_tile_size(self.parameters.param("Tile Size").value())

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

    def get_patch_view(
        self, anim: image.AnnotatedImage, patch_center=None, annotation_id=None
    ):
        if patch_center is not None:
            view: image.View = anim.get_view(
                center=[patch_center[0], patch_center[1]],
                level=self.level,
                size_on_level=self.tile_size,
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
        seg = tiles_processing(test_image, tile_fcn, tile_spacing=self.tile_size)

        if show:
            centers = tile_centers(test_image.shape, tile_spacing=self.tile_size)
            import skimage.color

            plt.imshow(skimage.color.label2rgb(seg, test_image))
            x, y = list(zip(*centers))
            plt.plot(x, y, "xy")
            # view.plot_points()
        return seg


def texture_glcm_features(img, levels):
    import skimage.feature.texture

    # levels =
    # if distances is None:
    distances = [1]
    # if angles is None:
    angles = [0, np.pi / 2]

    P = skimage.feature.greycomatrix(
        (img * (levels - 1)).astype(np.uint8),
        distances,
        angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    en = skimage.feature.texture.greycoprops(P, prop="energy")
    # dissimilarity = skimage.feature.texture.greycoprops(P, prop="dissimilarity")
    homogeneity = skimage.feature.texture.greycoprops(P, prop="homogeneity")
    correlation = skimage.feature.texture.greycoprops(P, prop="correlation")
    return np.array([np.mean(en), np.mean(homogeneity), np.mean(correlation)])


def texture_energy(img):
    import skimage.feature.texture

    P = skimage.feature.greycomatrix(
        (img * 31).astype(np.uint8),
        [1],
        [0, np.pi / 2],
        levels=32,
        symmetric=True,
        normed=True,
    )
    en = skimage.feature.texture.greycoprops(P, prop="energy")
    return np.mean(en) * 100
