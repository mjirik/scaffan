# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger

# import scaffan
# import io3d # just to get data
import scaffan.image as scim
from typing import List, Union
from pathlib import Path

# import sklearn.cluster
# import sklearn.naive_bayes
# import sklearn.svm
from scaffan.image import View, annoatation_px_to_mm
from sklearn.externals import joblib
from scipy.ndimage import gaussian_filter
import skimage
from sklearn.naive_bayes import GaussianNB
import numpy as np
from skimage.feature import peak_local_max
import skimage.filters
from skimage.morphology import disk
import scipy.ndimage
import matplotlib.pyplot as plt
from pyqtgraph.parametertree import Parameter, ParameterTree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import exsu
from exsu.report import Report
from .image import AnnotatedImage
from . import texture


class ScanSegmentation:
    def __init__(
        self,
        report: Report = None,
        pname="Scan Segmentation",
        ptype="bool",
        pvalue=True,
        ptip="Run analysis of whole slide before all other processing is perfomed",
    ):
        """

        An alternative features computation can be done by setting callback function
        self.alternative_get_features = fcn(seg:SlideSegmentation, view:View) -> ndarray:

        :param report:
        :param ptype: group or bool
        """
        # self._inner_texture:texture.GLCMTextureMeasurement = texture.GLCMTextureMeasurement(
        #     "slide_segmentation", texture_label="slide_segmentation", report_severity_offset=-30,
        #     glcm_levels=128
        # )
        params = [
            # {
            #     "name": "Tile Size",
            #     "type": "int",
            #     "value" : 128
            # },
            {
                "name": "Working Resolution",
                "type": "float",
                "value": 0.00001,  # 0.01 mm
                "suffix": "m",
                "siPrefix": True,
                "tip": "Resolution used for segmentation processing. "
                + "Real working resolution will be selected according to image levels.",
            },
            {
                "name": "Working Tile Size",
                "type": "int",
                "value": 256,
                "suffix": "px",
                "siPrefix": False,
                "tip": "Image is processed tile by tile. This value defines size of the tile",
            },
            {
                "name": "Run Training",
                "type": "bool",
                "value": False,
                "tip": "Use annotated image to train classifier."
                + "Red area is extra-lobular tissue."
                + "Black area is intra-lobular tissue."
                + "Magenta area is empty part of the image.",
            },
            {
                "name": "Load Default Classifier",
                "type": "bool",
                "value": False,
                "tip": "Load default classifier before training.",
            },
            {
                "name": "Clean Before Training",
                "type": "bool",
                "value": False,
                "tip": "Reset classifier before training.",
            },
            {
                "name": "Training Weight",
                "type": "float",
                "value": 1,
                # "suffix": "px",
                "siPrefix": False,
                "tip": "Weight of training samples given in actual image",
            },
            {
                "name": "Lobulus Number",
                "type": "int",
                "value": 5,
                # "suffix": "px",
                "siPrefix": False,
                "tip": "Number of lobuluses automatically selected.",
            },
            {
                "name": "Annotation Radius",
                "type": "float",
                "value": 0.00015,  # 0.1 mm
                "suffix": "m",
                "siPrefix": True,
                "tip": "Automatic annotation radius used when the automatic lobulus selection is prefered ",
            },
            # self._inner_texture.parameters,
        ]

        self.parameters = Parameter.create(
            name=pname,
            type=ptype,
            value=pvalue,
            tip=ptip,
            children=params,
            expanded=False,
        )
        if report is None:
            report = Report()
            report.save = False
            report.show = False
        self.report: Report = report
        self.anim: AnnotatedImage = None
        self.tile_size = None
        self.level = None
        self.tiles: List[List["View"]] = None
        # self._clf_object = SVC
        # self._clf_params = dict(gamma=2, C=1)
        self._clf_object = GaussianNB
        self._clf_params = {}
        # self._clf_object = DecisionTreeClassifier # no partial fit :-(
        # self._clf_params = dict(max_depth=5)

        #         self.clf = sklearn.svm.SVC(gamma='scale')
        # KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=2, C=1),
        # #     GaussianProcessClassifier(1.0 * RBF(1.0)),
        # self.clf_fn = DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        # QuadraticDiscriminantAnalysis(),
        # GaussianNB(),
        # self.clf = GaussianNB()
        self.clf = self._clf_object(**self._clf_params)
        self.clf_fn: Path = Path(Path(__file__).parent / "segmentation_model.pkl")
        self.clf_default_fn: Path = Path(
            Path(__file__).parent / "segmentation_model_default.pkl"
        )
        if self.clf_fn.exists():
            logger.debug(f"Reading classifier from {str(self.clf_fn)}")
            self.clf = joblib.load(self.clf_fn)
        self.predicted_tiles = None
        # self.output_label_fn = "label.png"
        # self.output_raster_fn = "image.png"
        self.devel_imcrop = None
        #         self.devel_imcrop = np.array([20000, 15000])
        self.full_output_image = None
        self.full_raster_image = None
        self.used_pixelsize_mm = None
        self.ann_biggest_ids = []
        self.empty_area_mm = None
        self.septum_area_mm = None
        self.sinusoidal_area_mm = None
        self.alternative_get_features = None

    def init(self, anim: scim.AnnotatedImage):
        self.anim = anim
        # self.anim = scim.AnnotatedImage(str(fn))
        self.level = self._find_best_level()
        self.tiles = None
        self.predicted_tiles = None
        self.tile_size = None
        self.ann_biggest_ids = []
        self.make_tiles()

    def train_classifier(self, pixels=None, y=None, sample_weight: float = None):
        logger.debug("start training")

        if pixels is None:
            pixels, y = self.prepare_training_pixels()
        if sample_weight is None:
            sample_weight = float(self.parameters.param("Training Weight").value())
        sample_weight = [sample_weight] * len(y)

        if bool(self.parameters.param("Clean Before Training").value()):
            self.clf = self._clf_object(**self._clf_params)
            # self.clf = GaussianNB()
            logger.debug(f"cleaning the classifier {self.clf}")
            self.clf.fit(pixels, y=y, sample_weight=sample_weight)
        else:
            self.clf.partial_fit(pixels, y=y, sample_weight=sample_weight)
        logger.debug("training finished")

    def save_classifier(self):
        logger.debug(f"save clf to {self.clf_fn}")
        joblib.dump(self.clf, self.clf_fn)

    def prepare_training_pixels(self):
        """
        Use annotated image to train classifier.
        Red area is extra-lobular tissue.
        Black area is intra-lobular tissue.
        Magenta area is empty part of the image.
        """
        pixels0 = self._get_pixels("#FF00FF")  # empty
        pixels1 = self._get_pixels("#000000")  # black
        pixels2 = self._get_pixels("#FF0000")  # extra lobula
        labels0 = np.ones([pixels0.shape[0]]) * 0
        labels1 = np.ones([pixels1.shape[0]]) * 1
        labels2 = np.ones([pixels2.shape[0]]) * 2
        pixels = np.concatenate([pixels0, pixels1, pixels2])
        y = np.concatenate([labels0, labels1, labels2])

        return pixels, y

    def _get_pixels(self, color: str):
        """
        Use outer annotation with defined color and removed holes to
        extract features in pixels.
        """
        outer_ids, holes_ids = self.anim.select_just_outer_annotations(color)
        views = self.anim.get_views(outer_ids, level=self.level)
        pixels_list = []
        for id1, id2, view_ann in zip(outer_ids, holes_ids, views):
            ann_raster = view_ann.get_annotation_raster(id1, holes_ids=id2)
            #     ann_raster1 = view_ann.get_annotation_region_raster(id1)
            #     if len(id2) == 0:
            #         ann_raster = ann_raster1
            #     else:
            #         ann_raster2 = view_ann.get_annotation_region_raster(id2[0])
            #         ann_raster = ann_raster1 ^ ann_raster2

            #             plt.figure()
            #             plt.imshow(ann_raster)
            #             plt.show()
            img = self._get_features(view_ann)
            pixels = img[ann_raster]
            pixels_list.append(pixels)
        pixels_all = np.concatenate(pixels_list, axis=0)
        return pixels_all

    def _get_features(self, view: View, debug_return=False) -> np.ndarray:
        """
        Three colors and one gaussian smooth reg channel.
        An alternative features computation can be done by setting callback function
        self.alternative_get_features = fcn(seg:SlideSegmentation, view:View) -> ndarray:

        img_sob: gaussian blure applied on gradient sobel operator give information about texture richness in neighborhood

        """
        img = view.get_region_image(as_gray=False)
        img_gauss2 = gaussian_filter(img[:, :, 0], 2)
        img_gauss5 = gaussian_filter(img[:, :, 0], 5)

        img = np.copy(img)
        nfeatures = 9
        # nfeatures = 12 #GLCM
        imgout = np.zeros([img.shape[0], img.shape[1], nfeatures], dtype=np.uint8)
        img_just_sob = skimage.filters.sobel(img[:, :, 0])
        # print(f"just_sob {img_just_sob.dtype} stats: {stats.describe(img_just_sob[:], axis=None)}")
        #         img_sob = (np.abs(img_just_sob) * 255).astype(np.uint8)
        img_sob = (np.abs(img_just_sob) * 255).astype(np.uint8)
        # print(f"sob {img_sob.dtype} stats: {stats.describe(img_sob[:], axis=None)}")
        img_sob_gauss2 = gaussian_filter(img_sob, 2)
        img_sob_gauss5 = gaussian_filter(img_sob, 5)
        img_sob_median = skimage.filters.median(
            (img_just_sob * 2000).astype(np.uint8), disk(10)
        )

        # GLCM
        # self._inner_texture.set_input_data(view=view, annotation_id=None, lobulus_segmentation=None)
        # self._inner_texture.run(recalculate_view=False)
        # glcm_features = (self._inner_texture.measured_features * 255).astype(np.uint8)
        # imgout[:, :, 9:12] = glcm_features[:, :, :3]

        imgout[:, :, :3] = img[:, :, :3]
        imgout[:, :, 3] = img_gauss2
        imgout[:, :, 4] = img_gauss5
        imgout[:, :, 5] = img_sob
        imgout[:, :, 6] = img_sob_gauss2
        imgout[:, :, 7] = img_sob_gauss5
        imgout[:, :, 8] = img_sob_median

        if debug_return:
            return imgout, [img_sob]
        return imgout

    def _find_best_level(self):
        pixelsize_mm = np.array(
            [float(self.parameters.param("Working Resolution").value()) * 1000] * 2
        )
        logger.debug(f"wanted pixelsize mm={pixelsize_mm}")
        error = None
        closest_i = None
        best_pxsz = None
        for i, pxsz in enumerate(self.anim.level_pixelsize):
            err = np.linalg.norm(pixelsize_mm - pxsz)
            if error is None:
                error = err
                closest_i = i
                best_pxsz = pxsz
            else:
                if err < error:
                    error = err
                    closest_i = i
                    best_pxsz = pxsz
        self.used_pixelsize_mm = best_pxsz
        logger.debug(f"real pixelsize mm={best_pxsz}")

        return closest_i

    def _get_tiles_parameters(self):
        height0 = self.anim.openslide.properties["openslide.level[0].height"]
        width0 = self.anim.openslide.properties["openslide.level[0].width"]

        imsize = np.array([int(width0), int(height0)])
        if self.devel_imcrop is not None:
            imsize = self.devel_imcrop

        tile_size_on_level = np.array(self.tile_size)
        downsamples = self.anim.openslide.level_downsamples[self.level]
        imsize_on_level = imsize / downsamples
        tile_size_on_level0 = tile_size_on_level * downsamples
        return (
            imsize.astype(np.int),
            tile_size_on_level0.astype(np.int),
            tile_size_on_level,
            imsize_on_level,
        )

    def make_tiles(self):
        sz = int(self.parameters.param("Working Tile Size").value())
        self.tile_size = [sz, sz]
        (
            imsize,
            size_on_level0,
            size_on_level,
            imsize_on_level,
        ) = self._get_tiles_parameters()
        self.tiles = []

        for x0 in range(0, int(imsize[0]), int(size_on_level0[0])):
            column_tiles = []

            for y0 in range(0, int(imsize[1]), int(size_on_level0[1])):
                logger.trace(f"processing tile {x0}, {y0}")
                view = self.anim.get_view(
                    location=(x0, y0), size_on_level=size_on_level, level=self.level
                )
                column_tiles.append(view)

            self.tiles.append(column_tiles)

    def predict_on_view(self, view):
        image = self._get_features(view)
        fvs = image.reshape(-1, image.shape[2])
        #         print(f"fvs: {fvs[:10]}")
        predicted = self.clf.predict(fvs).astype(np.int)
        img_pred = predicted.reshape(image.shape[0], image.shape[1])
        return img_pred

    def predict_tiles(self):
        if self.tiles is None:
            self.make_tiles()

        logger.debug("predicting tiles")
        self.predicted_tiles = []
        for i, tile_view_col in enumerate(self.tiles):
            predicted_col = []
            for j, tile_view in enumerate(tile_view_col):
                # self._inner_texture.texture_label = f"slide_segmentation_{i},{j}"
                predicted_image = self.predict_on_view(tile_view)
                predicted_col.append(predicted_image)
            self.predicted_tiles.append(predicted_col)

    def predict(self):
        """
        predict tiles and compose everything together
        """
        logger.debug("predict")
        if self.predicted_tiles is None:
            self.predict_tiles()

        #         if self.predicted_tiles is None:
        #             self.predict_tiles()

        szx = len(self.tiles)
        szy = len(self.tiles[0])
        #         print(f"size x={szx} y={szy}")

        (
            imsize,
            tile_size_on_level0,
            tile_size_on_level,
            imsize_on_level,
        ) = self._get_tiles_parameters()
        output_image = np.zeros(self.tile_size * np.asarray([szy, szx]), dtype=int)
        logger.debug("composing predicted image")
        for iy, tile_column in enumerate(self.tiles):
            for ix, tile in enumerate(tile_column):
                output_image[
                    ix * self.tile_size[0] : (ix + 1) * self.tile_size[0],
                    iy * self.tile_size[1] : (iy + 1) * self.tile_size[1],
                ] = self.predicted_tiles[iy][ix]

        full_image = output_image[: int(imsize_on_level[1]), : int(imsize_on_level[0])]
        self.full_prefilter_image = full_image
        self.full_output_image = self._labeling_filtration(full_image)
        return self.full_output_image

    def _labeling_filtration(self, full_image):
        """
        smooth label 0 and label 1, keep label 2
        """
        logger.debug("labeling filtration")
        tmp_img = full_image.copy()
        tmp_img[full_image == 2] = 1
        import skimage.filters

        tmp_img = skimage.filters.gaussian(tmp_img.astype(np.float), sigma=4)

        tmp_img = (tmp_img > 0.5).astype(np.int)
        tmp_img[full_image == 2] = 2
        return tmp_img

    def get_raster_image(self, as_gray=False):
        if self.tiles is None:
            self.make_tiles()
        szx = len(self.tiles)
        szy = len(self.tiles[0])
        #         print(f"size x={szx} y={szy}")

        output_size = self.tile_size * np.asarray([szy, szx])
        if not as_gray:
            output_size = np.asarray([output_size[0], output_size[1], 3])

        (
            imsize,
            tile_size_on_level0,
            tile_size_on_level,
            imsize_on_level,
        ) = self._get_tiles_parameters()
        output_image = np.zeros(output_size, dtype=int)
        for iy, tile_column in enumerate(self.tiles):
            for ix, tile in enumerate(tile_column):
                output_image[
                    ix * self.tile_size[0] : (ix + 1) * self.tile_size[0],
                    iy * self.tile_size[1] : (iy + 1) * self.tile_size[1]
                    #                     int(x0):int(x0 + tile_size_on_level[0]),
                    #                     int(y0):int(y0 + tile_size_on_level[1])
                    #                 ] = self.tiles[ix][iy].get_region_image(as_gray=True)
                ] = self.tiles[iy][ix].get_region_image(as_gray=as_gray)[:, :, :3]
        #                 ] = self.predicted_tiles[iy][ix]

        full_image = output_image[: int(imsize_on_level[1]), : int(imsize_on_level[0])]
        self.full_raster_image = full_image
        return full_image

    def evaluate(self):
        logger.debug("evaluate")
        _, count = np.unique(self.full_output_image, return_counts=True)
        self.intralobular_ratio = count[1] / (count[1] + count[2])
        #         plt.figure(figsize=(10, 10))
        #         plt.imshow(self.full_output_image)
        self.report.imsave(
            "slice_label.png", self.full_output_image, level_skimage=20, level_npz=30
        )
        self.report.imsave(
            "slice_prefilter_label.png",
            self.full_prefilter_image,
            level=40,
            level_skimage=20,
            level_npz=30,
        )
        # plt.imsave(self.output_label_fn, self.full_output_image)

        #         plt.figure(figsize=(10, 10))
        img = self.get_raster_image(as_gray=False)
        #         plt.imshow(img)
        self.report.imsave(
            "slice_raster.png", img.astype(np.uint8), level_skimage=20, level_npz=30
        )
        logger.debug(f"real_pixel_size={self.used_pixelsize_mm}")
        self.empty_area_mm = np.prod(self.used_pixelsize_mm) * count[0]
        self.sinusoidal_area_mm = np.prod(self.used_pixelsize_mm) * count[1]
        self.septum_area_mm = np.prod(self.used_pixelsize_mm) * count[2]
        logger.debug(f"empty_area_mm={self.empty_area_mm}")
        self.report.set_persistent_cols(
            {
                "Scan Segmentation Empty Area [mm^2]": self.empty_area_mm,
                "Scan Segmentation Septum Area [mm^2]": self.septum_area_mm,
                "Scan Segmentation Sinusoidal Area [mm^2]": self.sinusoidal_area_mm,
                "Scan Segmentation Used Pixelsize [mm]": self.used_pixelsize_mm[0],
                "Scan Segmentation Classifier": str(self.clf),
            }
        )

    def _find_biggest_lobuli(self):
        """
        :param n_max: Number of points. All points are returned if set to negative values.
        """
        n_max = int(self.parameters.param("Lobulus Number").value())
        mask = self.full_output_image == 1
        dist = scipy.ndimage.morphology.distance_transform_edt(mask)
        self.dist = dist
        # report

        image_max = scipy.ndimage.maximum_filter(dist, size=20, mode="constant")
        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(dist, min_distance=20)
        point_dist = dist[list(zip(*coordinates))]
        # display(point_dist)
        # max_point_inds = point_dist.argsort()[-n_max:][::-1]
        max_point_inds = point_dist.argsort()[::-1][:n_max]
        max_points = coordinates[max_point_inds]
        self.centers_all = coordinates
        self.centers_max = max_points

        #     report
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(dist, cmap=plt.cm.gray)
        plt.autoscale(False)
        plt.plot(
            coordinates[:, 1],
            coordinates[:, 0],
            "g.",
            max_points[:, 1],
            max_points[:, 0],
            "ro",
        )
        plt.axis("off")
        self.report.savefig_and_show(
            "sinusoidal_tissue_local_centers.pdf", fig, level=55
        )

        return max_points

    def add_biggest_to_annotations(self):
        points_px = self._find_biggest_lobuli()
        view_corner = self.tiles[0][0]
        pts_glob_px = view_corner.coords_view_px_to_glob_px(
            points_px[:, 1], points_px[:, 0]
        )
        centers_px = list(zip(*pts_glob_px))
        r_mm = float(self.parameters.param("Annotation Radius").value()) * 1000
        # r_mm = 0.1
        t = np.linspace(0, 2 * np.pi, 30)

        logger.debug(f"Automatic selection centers_px={centers_px}")
        for center_px in centers_px:
            r_px = view_corner.mm_to_px(r_mm)
            #     print(f"r_px={r_px}")
            r_px_glob = view_corner.coords_view_px_to_glob_px(
                np.array([r_px[0]]), np.array([r_px[1]])
            )
            x_px = r_px_glob[0] * np.sin(t) + center_px[0]
            y_px = r_px_glob[1] * np.cos(t) + center_px[1]

            ann = {
                "title": "Automatic Selection",
                "x_px": x_px,
                "y_px": y_px,
                "color": "#00FF88",
                "details": "",
            }
            ann = annoatation_px_to_mm(self.anim.openslide, ann)
            newid = len(self.anim.annotations)
            self.anim.annotations.append(ann)
            self.ann_biggest_ids.append(newid)
        # self.ann_biggest_ids = new_ann_ids

    def run(self):
        logger.debug("run...")
        # GLCM
        # self._inner_texture.set_report(self.report)
        # self._inner_texture.add_cols_to_report = False

        if bool(self.parameters.param("Load Default Classifier").value()):
            if self.clf_default_fn.exists():
                logger.debug(
                    f"Reading default classifier from {str(self.clf_default_fn)}"
                )
                self.clf = joblib.load(self.clf_default_fn)
            else:
                logger.error("Default classifier not found")

        if bool(self.parameters.param("Run Training").value()):
            self.train_classifier()
            self.save_classifier()
        logger.debug("predict...")
        self.predict()
        logger.debug("evaluate...")
        self.evaluate()
