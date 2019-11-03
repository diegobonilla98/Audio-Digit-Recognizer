import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
import logging
logging.getLogger('tensorflow').disabled = True


mypath = "spectrograms/"
image_files = [cv2.imread("spectrograms/" + f, 0) for f in listdir(mypath) if isfile(join(mypath, f))]
number_labels = [idx for idx in range(10) for num in range(200)]

image_files, number_labels = shuffle(image_files, number_labels, random_state=0)

y = to_categorical(number_labels)
X = np.array(image_files).reshape((2000, 128, 128, 1)).astype('float32') / 255

# model
filters = [128, 256, 512]

model = Sequential()
for filter in filters:
    model.add(layers.SeparableConv2D(filter, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(X, y, epochs=10, batch_size=16)

