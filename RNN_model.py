from keras.models import Sequential
from keras import layers


def rnnModel():
    model = Sequential()

    model.add(layers.BatchNormalization(input_shape=(18262, 1)))
    # model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.4, return_sequences=True))
    model.add(layers.GRU(32, activation='relu', dropout=0.1, recurrent_dropout=0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model
