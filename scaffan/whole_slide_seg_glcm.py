from pyqtgraph.parametertree import Parameter
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




class WholeSlideGLCM:
    def __init__(
        self,
        report: Report = None,
        pname="GLCM Segmentation",
        ptype="group",
        pvalue=None,
        ptip="Gray Level Co-occurence Matrix texture features based segmentation parameters",
    ):
        params = [
            {
                "name": "Model Name",
                "type": "str",
                "value": "posenet_highLR",
                # "suffix": "px",
                # "siPrefix": False,
                "tip": "Name of the used model",
            },
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
        pass

    def init_segmentation(self):
        # model = PoseNet()  # nacteni architektury modelu
        # # model_path = path_to_script / "models/"  # cesta k ulozenym modelum
        # model_path = path_to_script  # cesta k ulozenym modelum
        # # model_name = 'posenet_highLR' #nazev konkretniho modelu, mozna by slo dat do parametru pri volani
        # model_name = str(self.parameters.param("Model Name").value())
        # serializers.load_npz(
        #     model_path / (model_name + ".model"), model
        # )  # nacteni modelu
        # self.model = model

        pass

    def predict_tile(self, view: image.View):
        """
        predict image
        :param view:
        :return:
        """

        model = self.model

        t0 = time.time()
        grayscale_image = view.get_region_image(as_gray=True)
        t1 = time.time()
        if self.report:
            label = "profile unet cumulative get_region_image time [s]"
            if label in self.report.actual_row:
                self.report.actual_row[label] += float(t1 - t0)
            else:
                self.report.actual_row[label] = float(t1 - t0)

        # Get parameter value
        # sample_weight = float(self.parameters.param("Example Float Param").value())
        t0 = time.time()
        grayscale_image = np.expand_dims(
            np.expand_dims(grayscale_image, axis=0), axis=0
        ).astype("float32")
        prediction = F.softmax(
            model(grayscale_image)
        )  # predikce, vraci obrazek o rozmerech 224x224pix s hodnotami 0, 1, 2
        prediction = np.argmax(prediction.array, axis=1).astype("uint8")
        prediction = prediction.squeeze(0)
        t1 = time.time()

        if self.report:
            label = "profile unet cumulative prediction time [s]"
            if label in self.report.actual_row:
                self.report.actual_row[label] += float(t1 - t0)
            else:
                self.report.actual_row[label] = float(t1 - t0)
        return prediction
        # return (grayscale_image > 0.5).astype(np.uint8)
