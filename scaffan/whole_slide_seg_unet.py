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


class PoseNet(chainer.Chain):  # architektura PoseNetu
    def __init__(self, num_of_classes=3):
        super(PoseNet, self).__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(None, 16, 3, pad=1)
            self.l1b = L.BatchNormalization(16)
            self.l2 = L.Convolution2D(None, 32, 3, pad=1)
            self.l2b = L.BatchNormalization(32)
            self.l3 = L.Convolution2D(None, 32, 3, pad=1)
            self.l3b = L.BatchNormalization(32)
            self.l4 = L.Convolution2D(None, 64, 3, pad=1)
            self.l4b = L.BatchNormalization(64)

            self.l5 = L.Deconvolution2D(None, 64, 3, pad=1)
            self.l5b = L.BatchNormalization(64)
            self.l6 = L.Deconvolution2D(None, 32, 3, pad=1)
            self.l6b = L.BatchNormalization(32)
            self.l7 = L.Deconvolution2D(None, 32, 3, pad=1)
            self.l7b = L.BatchNormalization(32)
            self.l8 = L.Deconvolution2D(None, 16, 3, pad=1)
            self.l8b = L.BatchNormalization(16)

            self.lfin = L.Convolution2D(None, num_of_classes, 1)

    def __call__(self, x):
        h = self.l1(x)
        h = F.relu(self.l1b(h))
        h2 = self.l2(h)
        h2 = F.relu(self.l2b(h2))
        h3 = self.l3(h2)
        h3 = F.relu(self.l3b(h3))
        h4 = self.l4(h3)
        h4 = F.relu(self.l4b(h4))
        h4 = self.l5(h4)
        h4 = F.relu(self.l5b(h4))
        h4 = self.l6(h4)
        h4 = F.relu(self.l6b(h4))
        h4 = F.add(h3, h4)
        h4 = self.l7(h4)
        h4 = F.relu(self.l7b(h4))
        h4 = F.add(h2, h4)
        h4 = self.l8(h4)
        h4 = F.relu(self.l8b(h4))
        h4 = F.add(h, h4)

        return self.lfin(h4)


class WholeSlideSegmentationUNet:
    def __init__(
        self,
        report: Report = None,
        pname="U-Net",
        ptype="group",
        pvalue=None,
        ptip="CNN segmentation parameters",
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
        model = PoseNet()  # nacteni architektury modelu
        model_path = path_to_script / "models/"  # cesta k ulozenym modelum
        # model_name = 'posenet_highLR' #nazev konkretniho modelu, mozna by slo dat do parametru pri volani
        model_name = str(self.parameters.param("Model Name").value())
        serializers.load_npz(
            model_path / (model_name + ".model"), model
        )  # nacteni modelu
        self.model = model

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
