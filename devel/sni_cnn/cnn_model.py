import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy import load
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

TRAIN_DATA_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\train_data.npy"
TRAIN_LABELS_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\train_labels.npy"
TEST_DATA_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\test_data.npy"
TEST_LABELS_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\test_labels.npy"

VERSION = "1"
EXPORT_PATH = "D:\\FAV\\Scaffold\\export\\v" + VERSION + ".h5"

DISPLAY_SIZE = 80

BATCH_SIZE = 64
EPOCHS = 10


def compute_sample_weights(labels, hom):
    data_count = labels.shape[0]
    sample_weights = np.zeros((data_count,))

    lbls = labels.reshape((data_count,))

    label_values = list(set(lbls))
    label_value_counts = np.zeros((len(label_values),))
    label_value_weights = np.zeros((len(label_values),))

    for i, label_value in enumerate(label_values):
        for j in range(data_count):
            if abs(lbls[j] - label_value) < 0.001:
                label_value_counts[i] = label_value_counts[i] + 1

    for i in range(len(label_values)):
        label_value_weights[i] = 1 / (label_value_counts[i] / data_count)

    for i in range(data_count):
        for j, label_value in enumerate(label_values):
            if abs(lbls[i] - label_value) < 0.001:
                sample_weights[i] = hom[i] * label_value_weights[j]

    return sample_weights


def load_data():
    train_data = load(TRAIN_DATA_SAVE_FILE)
    train_labels = load(TRAIN_LABELS_SAVE_FILE)
    train_data_count = train_labels.shape[-1]

    train_hom = train_labels[0, :]
    train_sni = train_labels[1, :].reshape(train_data_count, 1)

    train_data = train_data.swapaxes(0, 2)
    train_data = train_data.reshape((train_data_count, DISPLAY_SIZE, DISPLAY_SIZE, 1))

    test_data = load(TEST_DATA_SAVE_FILE)
    test_labels = load(TEST_LABELS_SAVE_FILE)
    test_data_count = test_labels.shape[-1]

    test_hom = test_labels[0, :]
    test_sni = test_labels[1, :].reshape(test_data_count, 1)

    test_data = test_data.swapaxes(0, 2)
    test_data = test_data.reshape((test_data_count, DISPLAY_SIZE, DISPLAY_SIZE, 1))

    # sample_weights = compute_sample_weights(train_sni, train_hom)

    return train_data, train_sni, train_hom, test_data, test_sni, test_hom


def create_model():
    model = Sequential(
        [
            BatchNormalization(input_shape=(DISPLAY_SIZE, DISPLAY_SIZE, 1)),
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                input_shape=(DISPLAY_SIZE, DISPLAY_SIZE, 1),
            ),
            MaxPool2D((2, 2)),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPool2D((2, 2)),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPool2D((2, 2)),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPool2D((2, 2)),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.2),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(1, activation=None, use_bias=False),
        ]
    )

    return model


if __name__ == "__main__":
    from tensorflow import saved_model

    train_x, train_y, train_weights, test_x, test_y, test_weights = load_data()

    assert not np.any(np.isnan(train_x))

    assert not np.any(np.isnan(train_y))

    model = create_model()
    model.summary()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )

    history = model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=train_weights,
        validation_data=(test_x, test_y, test_weights),
    )

    model.save(EXPORT_PATH)

    matplotlib.use("TkAgg")
    plt.figure()
    plt.plot(history.history["mean_squared_error"], label="MSE")
    plt.plot(history.history["val_mean_squared_error"], label="val MSE")
    plt.xlabel("Epoch")
    plt.legend(["MSE", "val MSE"])
    plt.show()
