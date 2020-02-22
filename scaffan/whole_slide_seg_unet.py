from pyqtgraph.parametertree import Parameter
from . import image
from exsu import Report
import numpy as np
#
# The automatic test is in
# main_test.py: test_testing_slide_segmentation_clf_unet()

class WholeSlideSegmentationUNet:
    def __init__(
            self,
            report: Report = None,
            pname = "U-Net",
            ptype = "group",
            pvalue = None,
            ptip = "CNN segmentation parameters",

    ):

        # TODO Sem prosím všechny parametry.
        params = [

            {
                "name": "Example Integer Param",
                "type": "int",
                "value": 256,
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


    def init_segmentation(self):
        # TODO Tady si Ivane nandej, co je třeba udělat jednou před segmentací. Počítam nějaké načtení ze souboru atd.
        pass

    def predict_tile(self, view:image.View):
        """
        predict image
        :param view:
        :return:
        """
        # TODO tohle se volá pro každou dlaždici
        grayscale_image = view.get_region_image(as_gray=True)
        # Get parameter value
        sample_weight = float(self.parameters.param("Example Float Param").value())

        return (grayscale_image > 0.5).astype(np.uint8)
