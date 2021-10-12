from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from numpy import load
from matplotlib import pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
import numpy as np
from tensorflow import saved_model

TRAIN_DATA_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\train_data.npy'
TRAIN_LABELS_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\train_labels.npy'
TEST_DATA_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\test_data.npy'
TEST_LABELS_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\test_labels.npy'

DISPLAY_SIZE = 80

BATCH_SIZE = 64
EPOCHS = 10

def categorize_labels(labels):
    data_count = labels.shape[0]
    sample_weights = np.zeros((data_count,))

    lbls = labels.reshape((data_count,))


    label_values = list(set(lbls))

    print('Label categories: ' + str(label_values))

    # labels = np.zeros((labels.shape[0], len(label_values)))

    for i in range(data_count):
        for j, label_value in enumerate(label_values):
            if abs(lbls[i] - label_value) < 0.001:
                cat_label = np.zeros(len(label_values))
                cat_label[j] = 1
                labels[i, :] = j

    return labels.astype(int)


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

    train_sni = categorize_labels(train_sni)
    test_sni = categorize_labels(test_sni)

    return train_data, train_sni, train_hom, test_data, test_sni, test_hom


def create_model():
    model = Sequential([
        BatchNormalization(input_shape=(DISPLAY_SIZE, DISPLAY_SIZE, 1)),
        Conv2D(128, (3, 3), activation='relu', input_shape=(DISPLAY_SIZE, DISPLAY_SIZE, 1)),
        MaxPool2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(6, activation='softmax', use_bias=False)
    ])

    return model


if __name__ == '__main__':
    train_x, train_y, train_weights, test_x, test_y, test_weights = load_data()

    # train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # test_y.reshape((test_y.shape[0], test_y.shape[1], 1))

    # train_y = np.swapaxes(train_y, 0, 1)
    # test_y = np.swapaxes(test_y, 0, 1)

    model = create_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        sample_weight=train_weights,
                        validation_data=(test_x, test_y, test_weights))

    saved_model.save(model, "export_categorical/")

    matplotlib.use('TkAgg')
    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.xlabel('Epoch')
    plt.legend(['accuracy', 'val accuracy'])
    plt.show()

