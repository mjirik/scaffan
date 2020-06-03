from pyqtgraph.parametertree import Parameter
from loguru import logger
from . import image
from exsu import Report
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import os
from pathlib import Path
import time

#
# The automatic test is in
# main_test.py: test_testing_slide_segmentation_clf_unet()
path_to_script = Path(os.path.dirname(os.path.abspath(__file__)))
# path_to_scaffan = Path(os.path.join(path_to_script, ".."))




class LobuleQualityEstimationCNN:
    def __init__(
        self,
        report: Report = None,
        pname="Quality Estimation CNN",
        ptype="group",
        pvalue=None,
        ptip="CNN estimator of quality",
    ):
        params = [
            {
                "name": "Example Integer Param",
                "type": "int",
                "value": 224,
                "suffix": "px",
                "siPrefix": False,
                "tip": "Value defines size of something",
            },
            {
                "name": "Example Float Param",
                "type": "float",
                "value": 0.00006,
                "suffix": "m",
                "siPrefix": True,
                "tip": "Value defines size of something",
            },
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
        pass

    def init(self):
        # načtení architektury modelu
        # načtení parametrů modelu

        model_path = path_to_script / "nejlepsi.model"
        # self.model = model

    def set_input_data(
            self, view: image.View, annotation_id: int = None, lobulus_segmentation=None
    ):
        self.anim = view.anim
        self.annotation_id = annotation_id
        self.parent_view = view
        logger.trace(f"lobulus segmentation {lobulus_segmentation}")
        self.lobulus_segmentation = lobulus_segmentation

    def run(self):
        # Tady by bylo tělo algoritmu

        # výsledek uložený do proměnné sni_prediction
        sni_prediction = 0.65
        if self.report:
            label = "SNI prediction CNN"
            self.report.actual_row[label] = sni_prediction
        return sni_prediction
