from .filter import Filter
import numpy as np


class NearestNeighborFilter(Filter):
    """Filter which assign each pixel to the nearest centroid of the model."""

    def create_model(self):
        self.centroinds = []

    def train_model(self, img, seeds):
        # get all types of annotation
        ann_types = set(list(seeds.reshape(seeds.size)))
        # remove not-annotated type
        ann_types.remove(0)
        ann_types = list(ann_types)
        # empty centroids list
        self.centroids = []

        # compute centroid for each class
        for level in ann_types:
            self.centroids.append(np.sum(img[seeds == level], axis=0) / np.sum(seeds == level))

    def predict(self, img):
        # repeat each centroid to match input image shape
        centroid_maps = []
        for centroid in self.centroids:
            centroid_map = np.tile(centroid, img.shape[0] * img.shape[1]).reshape(img.shape)
            centroid_maps.append(centroid_map)

        # compute distance from each image pixel to each centroid
        centroid_distances = []
        for centroid_map in centroid_maps:
            centroid_distances.append(abs(img - centroid_map))

        # merge results to output filter mask
        centroid_distances = np.sum(centroid_distances, axis=-1)
        filter_mask = np.argmin(np.array(centroid_distances), axis=0)

        return filter_mask
