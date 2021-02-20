import unittest

import numpy as np
from matplotlib import pyplot as plt

from .nearest_neighbor_filter import NearestNeighborFilter
from .svm_filter import SvmFilter
from .nbg_filter import NaiveBayesFilter
from .dataset_creator import load_image
from .visualization import show_visual

models = [NearestNeighborFilter(), SvmFilter(), NaiveBayesFilter()]

class TestFilter(unittest.TestCase):
    """"""

    def model_test(self, model):
        """"""
        img = np.load('image.npy')
        seeds = np.load('seeds.npy')

        model.train_model(img, seeds)
        output = model.predict(img)
        model.save_model()
        model.load_model()

    def test_all_models(self):
        for model in models:
            self.model_test(model)

    def test_visual_comparsion(self):
        img = load_image()

        plt.figure()
        plt.imshow(img)
        plt.title("Original image")

        model_names = ['Nearest Neighbor', 'SVM', 'Naive Bayes Gaussian']

        for i in range(len(models)):
            show_visual(img, models[i], model_names[i])
