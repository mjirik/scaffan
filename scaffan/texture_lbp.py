# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for texrure analysis.
"""
import logging
logger = logging.getLogger(__name__)


import numpy as np
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
import matplotlib.pyplot as plt

# settings for LBP
radius = 2
n_points = 8 * radius
METHOD = 'uniform'


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    if type(refs) is dict:
        itms = refs.items()
    elif type(refs) is list:
        itms = refs
    else:
        ValueError("Wrong type for 'refs'")

    for name, ref in itms:
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name

def show_lbp(lbp):
    n_bins = int(lbp.max() + 1)
    plt.hist(lbp.ravel(), range=(0, n_bins), bins=n_bins, normed=True)
    plt.xlim(xmax=n_points + 2)

