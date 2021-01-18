import abc


class Filter:
    """Interface for image filter implementations"""

    def __init__(self):
        self.model = self.create_model()

    @abc.abstractmethod
    def create_model(self):
        """Create and initialize the classification model."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_model(self, data):
        """Train the model to fit the train data."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, image):
        """Use trained model for input image segmentation"""
        raise NotImplementedError
