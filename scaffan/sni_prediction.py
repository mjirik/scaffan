# /usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

from loguru import logger
import numpy as np
from pyqtgraph.parametertree import Parameter
import os.path
from pathlib import Path
import joblib
import sklearn

from exsu.report import Report
from . import image

path_to_script = Path(os.path.dirname(os.path.abspath(__file__)))


class SniPredictor:
    def __init__(
        self,
        report: Report = None,
        pname="SNI prediction",
        ptype="group",
        pvalue=None,
        ptip="Prediction of SNI based on pre-trained regression from texture features.",
    ):
        params = [
            # {
            #     "name": "Model Name",
            #     "type": "str",
            #     "value": "posenet_highLR",
            #     # "suffix": "px",
            #     # "siPrefix": False,
            #     "tip": "Name of the used model",
            # },
            # {
            #     "name": "Example Integer Param",
            #     "type": "int",
            #     "value": 224,
            #     "suffix": "px",
            #     "siPrefix": False,
            #     "tip": "Value defines size of something",
            # },
            # {
            #     "name": "Example Float Param",
            #     "type": "float",
            #     "value": 0.00006,
            #     "suffix": "m",
            #     "siPrefix": True,
            #     "tip": "Value defines size of something",
            # },
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
        # self.init_regressors()

        # def init_regressors(self):
        # model = PoseNet()  # nacteni architektury modelu
        self.area_regressor_path = path_to_script / "models/SNI_area_regressor.joblib"
        self.perpixel_regressor_path = (
            path_to_script / "models/SNI_per-pixel_regressor.joblib"
        )
        areg = joblib.load(self.area_regressor_path)
        ppreg = joblib.load(self.area_regressor_path)
        self.areg: sklearn.linear_model.LinearRegression = areg["regressor"]
        self.areg_features = areg["features"]

        self.ppreg: sklearn.linear_model.LinearRegression = ppreg["regressor"]
        self.ppreg_features = ppreg["features"]

    def predict_area(self, row: dict):
        logger.debug(f"SNI row keys {row.keys()}")

        X = np.array([[row[feature_name] for feature_name in self.areg_features]])
        sni_area = self.areg.predict(X)[0]
        if self.report is not None:
            self.report.add_cols_to_actual_row({"SNI area prediction": sni_area})
        logger.debug("SNI area prediction done for one lobuli")

    def predict_tile(self, view: image.View):
        """
        predict image
        :param view:
        :return:
        """
        pass

        # model = self.model
        #
        # grayscale_image = view.get_region_image(as_gray=True)
        #
        # # Get parameter value
        # # sample_weight = float(self.parameters.param("Example Float Param").value())
        # return (grayscale_image > 0.5).astype(np.uint8)
