from os import listdir
from os.path import isfile, join
from keras.optimizers import RMSprop
from RNN_model import rnnModel
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from model_checkpoint import saveModel
from conv_model import convModel
from keras.utils import to_categorical
from scipy.io import wavfile
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

logging.getLogger('tensorflow').disabled = True

mypath = "recordings"
audio_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

audio_0 = [f for f in audio_files if f[0] == '0']
audio_1 = [f for f in audio_files if f[0] == '1']
audio_2 = [f for f in audio_files if f[0] == '2']
audio_3 = [f for f in audio_files if f[0] == '3']
audio_4 = [f for f in audio_files if f[0] == '4']
audio_5 = [f for f in audio_files if f[0] == '5']
audio_6 = [f for f in audio_files if f[0] == '6']
audio_7 = [f for f in audio_files if f[0] == '7']
audio_8 = [f for f in audio_files if f[0] == '8']
audio_9 = [f for f in audio_files if f[0] == '9']

audios = [audio_0, audio_1, audio_2, audio_3, audio_4,
          audio_5, audio_6, audio_7, audio_8, audio_9]
audio_arr = []
for num, audio in enumerate(audios):
    for idx, wav in enumerate(audio):
        _, samples = wavfile.read('recordings/' + wav)
        audio_arr.append(samples)
audio_arr = pad_sequences(audio_arr)
audio_arr = np.array(audio_arr)
audio_arr = audio_arr.astype('float32')

# scaler = StandardScaler()
# scaler.fit(audio_arr)
# trainy = scaler.transform(audio_arr)

number_labels = [idx for idx in range(10) for num in range(200)]

audio_arr, number_labels = shuffle(audio_arr, number_labels, random_state=0)

number_labels = to_categorical(number_labels).astype('float32')
audio_arr = audio_arr.reshape(2000, 18262, 1)
print(audio_arr.shape)
print(number_labels.shape)


# conv_model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001, cooldown=1)
tensorboard = TensorBoard(log_dir='my_log_dir', histogram_freq=1)
callbacks = [tensorboard, reduce_lr]
conv_model = convModel()
rmsprop = RMSprop(lr=0.001)
conv_model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['acc'])
conv_model.fit(audio_arr, number_labels, epochs=30, batch_size=32,
               callbacks=callbacks, validation_split=0.1)

saveModel(conv_model)

