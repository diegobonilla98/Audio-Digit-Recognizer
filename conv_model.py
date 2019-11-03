from keras import layers
from keras.models import Sequential


def convModel(num_filter=7, filters=(32, 64)):
    model = Sequential()
    model.add(layers.BatchNormalization(input_shape=(18262, 1)))

    model.add(layers.Conv1D(filters[0], num_filter, activation='relu'))
    model.add(layers.Conv1D(filters[0], num_filter, activation='relu'))
    model.add(layers.MaxPooling1D(3))

    model.add(layers.Conv1D(filters[1], num_filter, activation='relu'))
    model.add(layers.Conv1D(filters[1], num_filter, activation='relu'))
    model.add(layers.MaxPooling1D(3))

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model
