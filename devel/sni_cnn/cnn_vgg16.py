import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy import load

TRAIN_DATA_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\train_data.npy"
TRAIN_LABELS_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\train_labels.npy"
TEST_DATA_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\test_data.npy"
TEST_LABELS_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\test_labels.npy"

VERSION = "_VGG16_2"
EXPORT_PATH = "D:\\FAV\\Scaffold\\export\\v" + VERSION + "/"

DISPLAY_SIZE = 80

BATCH_SIZE = 64
EPOCHS = 10


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

    return train_data, train_sni, train_hom, test_data, test_sni, test_hom


def create_model():
    from tensorflow import saved_model
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Conv2D(
                input_shape=(DISPLAY_SIZE, DISPLAY_SIZE, 1),
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
            ),
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(units=4096, activation="relu"),
            Dense(units=4096, activation="relu"),
            Dense(units=1, activation=None, use_bias=False),
        ]
    )

    return model


if __name__ == "__main__":
    from tensorflow import saved_model
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

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

    saved_model.save(model, EXPORT_PATH)

    matplotlib.use("TkAgg")
    plt.figure()
    plt.plot(history.history["mean_squared_error"], label="MSE")
    plt.plot(history.history["val_mean_squared_error"], label="val MSE")
    plt.xlabel("Epoch")
    plt.legend(["MSE", "val MSE"])
    plt.show()
