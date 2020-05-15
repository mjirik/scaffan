from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from numpy import load
from matplotlib import pyplot as plt
from tensorflow.keras import datasets

TRAIN_DATA_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\train_data.npy'
LABELS_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\labels.npy'
DISPLAY_SIZE = 80

BATCH_SIZE = 8
EPOCHS = 30


def load_data():
    train_data = load(TRAIN_DATA_SAVE_FILE)
    labels = load(LABELS_SAVE_FILE)
    data_count = labels.shape[-1]

    hom = labels[0, :]
    sni = labels[1, :].reshape(data_count, 1)

    return train_data.reshape((data_count, DISPLAY_SIZE, DISPLAY_SIZE, 1)), sni, hom


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(DISPLAY_SIZE, DISPLAY_SIZE, 1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation=None, use_bias=False))

    return model


if __name__ == '__main__':
    x, y, hom = load_data()

    print(x.shape)

    model = create_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    history = model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, sample_weight=hom)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("weights.h5")
    print("Saved model to disk")

    plt.figure()
    plt.plot(history.history['mean_squared_error'], label='mean_squared_error')
    plt.xlabel('Epoch')
    plt.legend(['MSE '])
    plt.show()

