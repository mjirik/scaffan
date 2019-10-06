# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import scaffan
import io3d # just to get data
import scaffan.image as scim
from typing import List
from pathlib import Path
import sklearn.cluster
import sklearn.naive_bayes
import sklearn.svm
from scaffan.image import View
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


class SlideSegmentation():
    def __init__(self):
        self.anim = None
        self.pixelsize_mm = [0.01, 0.01]
        self.tile_size = [256, 256]
        self.level = None
        self.tiles: List["View"] = None
        #         self.clf = sklearn.svm.SVC(gamma='scale')
        self.clf = GaussianNB()
        self.clf_fn = "standard_svm_model.pkl"
        self.clf = joblib.load(self.clf_fn)
        self.predicted_tiles = None
        self.output_label_fn = "label.png"
        self.output_raster_fn = "image.png"
        self.devel_imcrop = None
        #         self.devel_imcrop = np.array([20000, 15000])
        self.full_output_image = None
        self.full_raster_image = None

        pass

    def init(self, fn: Path):
        self.anim = scim.AnnotatedImage(fn)
        self.level = self._find_best_level()
        self.tiles = None
        #         self.predicted_tiles = None
        self.make_tiles()

    def train_svm_classifier(self, pixels=None, y=None):
        if pixels is None:
            pixels, y = self.prepare_training_pixels()

        self.clf.fit(pixels, y=y)

    def save_classifier(self):
        joblib.dump(self.clf, 'standard_svm_model.pkl')

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

    #     def _get_pixels(self, anim, color, n):
    #         ann_ids = anim.select_annotations_by_color(color)
    #         view = anim.get_views(ann_ids, level=self.level)[n]
    #         img_ann = view.get_annotation_region_raster(ann_ids[n])

    #         img = self._get_features(view)
    #         pixels = img[img_ann]

    #         return pixels, view

    def _get_features(self, view: View):
        """
        Three colors and one gaussian smooth reg channel.

        img_sob: gaussian blure applied on gradient sobel operator give information about texture richness in neighborhood

        """
        img = view.get_region_image(as_gray=False)
        img_gauss2 = gaussian_filter(img[:, :, 0], 2)
        img_gauss5 = gaussian_filter(img[:, :, 0], 5)

        img = np.copy(img)
        imgout = np.zeros([img.shape[0], img.shape[1], 8])
        img_sob = skimage.filters.sobel(img[:, :, 0] / 255)
        img_sob_gauss2 = gaussian_filter(img_sob, 2)
        img_sob_gauss5 = gaussian_filter(img_sob, 5)
        img_sob_median = skimage.filters.median(img_sob, disk(5))

        imgout[:, :, :3] = img[:, :, :3]
        imgout[:, :, 3] = img_gauss2
        imgout[:, :, 4] = img_gauss5
        imgout[:, :, 5] = img_sob_gauss2
        imgout[:, :, 6] = img_sob_gauss5
        imgout[:, :, 7] = img_sob_median
        return imgout

    def _find_best_level(self):
        error = None
        closest_i = None
        for i, pxsz in enumerate(self.anim.level_pixelsize):
            err = np.linalg.norm(self.pixelsize_mm - pxsz)
            if error is None:
                error = err
                closest_i = i
            else:
                if err < error:
                    error = err
                    closest_i = i

        return closest_i

    def _get_tiles_parameters(self):
        height0 = self.anim.openslide.properties['openslide.level[0].height']
        width0 = self.anim.openslide.properties['openslide.level[0].width']

        imsize = np.array([int(width0), int(height0)])
        if self.devel_imcrop is not None:
            imsize = self.devel_imcrop

        tile_size_on_level = np.array(self.tile_size)
        downsamples = self.anim.openslide.level_downsamples[self.level]
        imsize_on_level = imsize / downsamples
        tile_size_on_level0 = tile_size_on_level * downsamples
        return imsize.astype(np.int), tile_size_on_level0.astype(np.int), tile_size_on_level, imsize_on_level

    def make_tiles(self):
        imsize, size_on_level0, size_on_level, imsize_on_level = self._get_tiles_parameters()
        self.tiles = []

        for x0 in range(0, int(imsize[0]), int(size_on_level0[0])):
            column_tiles = []

            for y0 in range(0, int(imsize[1]), int(size_on_level0[1])):
                view = self.anim.get_view(location=(x0, y0), size_on_level=size_on_level, level=self.level)
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

        self.predicted_tiles = []
        for tile_view_col in self.tiles:
            predicted_col = []
            for tile_view in tile_view_col:
                predicted_image = self.predict_on_view(tile_view)
                predicted_col.append(predicted_image)
            self.predicted_tiles.append(predicted_col)

    def predict(self):
        """
        predict tiles and compose everything together
        """
        if self.predicted_tiles is None:
            self.predict_tiles()

        #         if self.predicted_tiles is None:
        #             self.predict_tiles()

        szx = len(self.tiles)
        szy = len(self.tiles[0])
        #         print(f"size x={szx} y={szy}")

        imsize, tile_size_on_level0, tile_size_on_level, imsize_on_level = self._get_tiles_parameters()
        output_image = np.zeros(self.tile_size * np.asarray([szy, szx]), dtype=int)
        for iy, tile_column in enumerate(self.tiles):
            for ix, tile in enumerate(tile_column):
                output_image[
                ix * self.tile_size[0]: (ix + 1) * self.tile_size[0],
                iy * self.tile_size[1]: (iy + 1) * self.tile_size[1]
                #                     int(x0):int(x0 + tile_size_on_level[0]),
                #                     int(y0):int(y0 + tile_size_on_level[1])
                #                 ] = self.tiles[ix][iy].get_region_image(as_gray=True)
                #                 ] = self.tiles[iy][ix].get_region_image(as_gray=True)
                ] = self.predicted_tiles[iy][ix]

        full_image = output_image[:int(imsize_on_level[1]), :int(imsize_on_level[0])]
        self.full_prefilter_image = full_image
        self.full_output_image = self._labeling_filtration(full_image)
        return self.full_output_image

    def _labeling_filtration(self, full_image):
        """
        smooth label 0 and label 1, keep label 2
        """
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

        imsize, tile_size_on_level0, tile_size_on_level, imsize_on_level = self._get_tiles_parameters()
        output_image = np.zeros(output_size, dtype=int)
        for iy, tile_column in enumerate(self.tiles):
            for ix, tile in enumerate(tile_column):
                output_image[
                ix * self.tile_size[0]: (ix + 1) * self.tile_size[0],
                iy * self.tile_size[1]: (iy + 1) * self.tile_size[1]
                #                     int(x0):int(x0 + tile_size_on_level[0]),
                #                     int(y0):int(y0 + tile_size_on_level[1])
                #                 ] = self.tiles[ix][iy].get_region_image(as_gray=True)
                ] = self.tiles[iy][ix].get_region_image(as_gray=as_gray)[:, :, :3]
        #                 ] = self.predicted_tiles[iy][ix]

        full_image = output_image[:int(imsize_on_level[1]), :int(imsize_on_level[0])]
        self.full_raster_image = full_image
        return full_image

    def evaluate(self):
        _, count = np.unique(self.full_output_image, return_counts=True)
        self.intralobular_ratio = count[1] / (count[1] + count[2])
        #         plt.figure(figsize=(10, 10))
        #         plt.imshow(self.full_output_image)
        plt.imsave(self.output_label_fn, self.full_output_image)

        #         plt.figure(figsize=(10, 10))
        img = self.get_raster_image(as_gray=False)
        #         plt.imshow(img)
        plt.imsave(self.output_raster_fn, img.astype(np.uint8))

    def find_biggest_lobuli(self, n_max: int = 5):
        """
        :param n_max: Number of points. All points are returned if set to 0.
        """
        mask = self.full_output_image == 1
        dist = scipy.ndimage.morphology.distance_transform_edt(mask)
        self.dist = dist
        # report

        image_max = scipy.ndimage.maximum_filter(dist, size=20, mode='constant')
        # Comparison between image_max and im to find the coordinates of local maxima
        coordinates = peak_local_max(dist, min_distance=20)
        point_dist = dist[list(zip(*coordinates))]
        # display(point_dist)
        max_point_inds = point_dist.argsort()[-n_max:][::-1]
        max_points = coordinates[max_point_inds]
        self.centers_all = coordinates
        self.centers_max = max_points

        #     report
        plt.figure(figsize=(10, 10))
        plt.imshow(dist, cmap=plt.cm.gray)
        plt.autoscale(False)
        plt.plot(coordinates[:, 1], coordinates[:, 0], 'g.')
        plt.plot(max_points[:, 1], max_points[:, 0], 'ro')
        plt.axis('off')

        return max_points

    def run(self):
        self.predict()
        self.evaluate()


