import abc


class Filter:
    """Interface for image filter implementations"""

    def __init__(self):
        self.class_count = 3
        self.model = self.create_model()

    @abc.abstractmethod
    def create_model(self):
        """Create and initialize the classification model."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_model(self, img, seeds):
        """Train the model to fit the train data.
        :parameter img: input image
        :parameter seeds: seeds (sparse labels)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, img):
        """Use trained model for input image segmentation
        :parameter img: input image
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, file_name):
        """Load model parameters from file.
        :parameter file_name: file name
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, file_name):
        """Save model parameters from file.
        :parameter file_name: file name
        """
        raise NotImplementedError
