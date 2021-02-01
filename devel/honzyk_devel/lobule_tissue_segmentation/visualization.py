"""Script for image filer visualization"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv

from .dataset_creator import load_image, hue_to_continuous_2d
from .nearest_neighbor_filter import NearestNeighborFilter
from .svm_filter import SvmFilter


def show_visual(img, model, model_name):
    model.load_model()

    img_hhsv = rgb2hsv(img)
    img_hhsv = hue_to_continuous_2d(img_hhsv)

    neigh_result = model.predict(img_hhsv)
    svm_result = model.predict(img_hhsv)

    img = img.astype('float32') / 255.0

    plt.figure()
    plt.imshow(neigh_result, cmap='brg')
    plt.title("3 different tissue - RGB - " + model_name)

    # Black
    plt.figure()
    plt.imshow(img * np.stack([neigh_result == 0, neigh_result == 0, neigh_result == 0], axis=-1) + np.stack(
        [neigh_result != 0, neigh_result != 0, neigh_result != 0],
        axis=-1).astype('float32'))
    plt.title("Black tissue - " + model_name)

    # White
    plt.figure()
    plt.imshow(img * np.stack([neigh_result == 1, neigh_result == 1, neigh_result == 1], axis=-1))
    plt.title("White tissue - " + model_name)

    # Brown
    plt.figure()
    plt.imshow(img * np.stack([neigh_result == 2, neigh_result == 2, neigh_result == 2], axis=-1))
    plt.title("Brown tissue - " + model_name)

    plt.show()
