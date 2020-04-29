from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from numpy import load
from matplotlib import pyplot as plt
from tensorflow.keras import datasets

IMAGES_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\images.npy'
HOMOS_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\homos.npy'


def load_data():
    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    images = load(IMAGES_SAVE_FILE)
    homos = load(HOMOS_SAVE_FILE)
    return images.reshape((288, 300, 300, 1)), homos.reshape(288, 5)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax', use_bias=False)) # 1 neuron bez aktivace

    return model


if __name__ == '__main__':
    x, y = load_data()

    print(x.shape)

    model = create_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # mean square error
                  metrics=['accuracy'])

    history = model.fit(x, y, epochs=5, batch_size=5)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
