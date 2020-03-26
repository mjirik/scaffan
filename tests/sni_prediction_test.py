# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import pytest
import scaffan
import io3d  # just to get data
import scaffan.image as scim
from typing import List
import exsu
import unittest
from scaffan import sni_prediction
import numpy as np


def test_sni_prediction():
    # report = exsu.Report()
    sni_predictor = sni_prediction.SniPredictor()
    report = sni_predictor.report

    assert "SNI area prediction" not in report.actual_row
    sni_predictor.predict_area({
        #     "Branch number", "Skeleton length",
        #                                   'GLCM Correlation',
        #        'GLCM Correlation p10', 'GLCM Correlation p25', 'GLCM Correlation p50',
        #        'GLCM Correlation p75', 'GLCM Correlation p90',
        'GLCM Correlation var': 0.013,
        #        'GLCM Energy', 'GLCM Energy p10', 'GLCM Energy p25', 'GLCM Energy p50',
        #        'GLCM Energy p75',
        'GLCM Energy p90': 0.73,
        #     'GLCM Energy var',
        #        'GLCM Homogenity', 'GLCM Homogenity p10', 'GLCM Homogenity p25',
        #        'GLCM Homogenity p50', 'GLCM Homogenity p75',
        'GLCM Homogenity p90': 0.90
        #        'GLCM Homogenity var',
        #         'Lobulus Boundary Compactness',
        })
    assert "SNI area prediction" in report.actual_row
    assert type(report.actual_row["SNI area prediction"]) in [int, float, np.float64, np.float]


