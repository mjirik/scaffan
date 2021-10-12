import pickle as plk

from sklearn import svm

from .filter import Filter


class SvmFilter(Filter):
    """Filter which assign each pixel to the nearest centroid of the model."""

    def create_model(self):
        return svm.LinearSVC()

    def train_model(self, img, seeds):
        # bitmap to string of pixels
        img = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
        seeds = seeds.reshape((seeds.shape[0] * seeds.shape[1]))

        # remove not labeled pixels
        img = img[seeds > 0.0]
        seeds = seeds[seeds > 0.0]

        # train model
        self.model.fit(img, seeds)

    def predict(self, img):
        orig_shape = img.shape

        # bitmap to string of pixels
        img = img.reshape((orig_shape[0] * orig_shape[1], orig_shape[2]))

        filter_mask = self.model.predict(img)
        filter_mask = filter_mask.reshape(orig_shape[0], orig_shape[1])

        return filter_mask

    def load_model(self, file_name='svm_filter.npy'):
        with open(file_name, 'rb') as file:
            self.model = plk.load(file)

    def save_model(self, file_name='svm_filter.npy'):
        with open(file_name, 'wb') as file:
            plk.dump(self.model, file)
