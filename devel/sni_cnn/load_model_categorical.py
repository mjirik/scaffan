import matplotlib
from matplotlib import pyplot as plt
from numpy import load
from tensorflow.keras.models import load_model
import random
import scaffan.scaffan.image as scim
import matplotlib.patches as patches
from training_data_generator import load_lobulus
from training_data_generator import cut_the_image
from training_data_generator import shrink_image
import numpy as np
from statistics import mean

TRAIN_DATA_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\train_data.npy"
TRAIN_LABELS_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\train_labels.npy"
TEST_DATA_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\test_data.npy"
TEST_LABELS_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\test_labels.npy"
DISPLAY_SIZE = 80
CUT_SIZE = 0.2  # Size of training data [mm]


def compute_MSE_of_mean_value(y):
    y = y.reshape(len(y))
    mean = y.mean()
    mse = 0
    for i in range(len(y)):
        mse = mse + (y[i] - mean) * (y[i] - mean)

    mse = mse / len(y)
    return mse


def manual_validation_1():
    x_test = load(TRAIN_DATA_SAVE_FILE)
    labels = load(TRAIN_LABELS_SAVE_FILE)
    data_count = labels.shape[-1]

    x_test = x_test.swapaxes(0, 2)
    x_test = x_test.reshape((data_count, DISPLAY_SIZE, DISPLAY_SIZE, 1))

    y_test = labels[1, :].reshape(data_count, 1)

    data_count = y_test.shape[0]
    loaded_model = load_model("export_categorical/")

    score = loaded_model.evaluate(x_test, y_test, verbose=0)

    print(
        "MSE for guessing mean value of labels: "
        + str(compute_MSE_of_mean_value(y_test))
    )
    print("MSE for CNN: " + str(score[1]))

    while True:
        print("Close window to see the next one or stop the program.")
        test_index = random.randrange(data_count)
        testing_image = x_test[test_index, :, :, 0]

        prediction = loaded_model.predict(
            testing_image.reshape(1, DISPLAY_SIZE, DISPLAY_SIZE, 1), verbose=0
        )

        print("Guess:" + str(np.round(prediction, 3)))
        print("Real value:" + str(y_test[test_index]))

        plt.imshow(testing_image)
        plt.gray()
        plt.show()


def evaluate_annotation(file_path, ann_id, visual=False):
    loaded_model = load_model("export_categorical/")
    anim = scim.AnnotatedImage(file_path)
    lobulus = load_lobulus(anim, ann_id)
    cuts = cut_the_image(lobulus, False)
    crop_size = int((1 / lobulus.view.region_pixelsize[0]) * CUT_SIZE)

    evaluations = []

    for i, cut_point in enumerate(cuts):
        image_crop = lobulus.image[
            cut_point[0] : cut_point[0] + crop_size,
            cut_point[1] : cut_point[1] + crop_size,
        ]

        image_crop = shrink_image(image_crop)

        prediction = loaded_model.predict(
            image_crop.reshape(1, DISPLAY_SIZE, DISPLAY_SIZE, 1), verbose=0
        )
        evaluations.append(2 * np.float(prediction))

    fig, ax = plt.subplots(1)

    ax.imshow(lobulus.lobulus_mask.astype(float) * lobulus.image)

    for i, point in enumerate(cuts):
        rect = patches.Rectangle(
            (point[0], point[1]),
            crop_size,
            crop_size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(point[0] + 10, point[1] + 75, str(round(evaluations[i], 3)))

    mean_evaluation = mean(evaluations)
    plt.title("Mean SNI - CNN evaluation: " + str(np.round(mean_evaluation, 3)))
    plt.show()


if __name__ == "__main__":
    matplotlib.use("TkAgg")

    manual_validation_1()
