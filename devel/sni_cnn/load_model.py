from keras.models import model_from_json
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import load
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

x_test = load(TRAIN_DATA_SAVE_FILE)
labels = load(LABELS_SAVE_FILE)
data_count = labels.shape[-1]

x_test = x_test.reshape((data_count, DISPLAY_SIZE, DISPLAY_SIZE, 1))

# hom = labels[0, :]
y_test = labels[1, :].reshape(data_count, 1)

print(x_test.shape)

loaded_model = Sequential()
loaded_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(DISPLAY_SIZE, DISPLAY_SIZE, 1)))
loaded_model.add(MaxPool2D((2, 2)))
loaded_model.add(Conv2D(64, (3, 3), activation='relu'))
loaded_model.add(MaxPool2D((2, 2)))
loaded_model.add(Conv2D(64, (3, 3), activation='relu'))
loaded_model.add(Flatten())
loaded_model.add(Dense(64, activation='relu'))
loaded_model.add(Dense(1, activation=None, use_bias=False))

loaded_model.load_weights("weights.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)

while True:
    print('Zadejte prosím index trénovacích dat (0-51)')
    test_index = int(input())
    testing_image = x_test[test_index].reshape(1, DISPLAY_SIZE, DISPLAY_SIZE, 1)
    testing_image_show = testing_image.reshape(DISPLAY_SIZE, DISPLAY_SIZE)

    prediction = loaded_model.predict(testing_image, verbose=0)

    print('Odhad hodnoty:' + str(prediction))
    print('Skutečná hodnota:' + str(prediction))

    plt.imshow(testing_image_show)
    plt.gray()
    plt.show()