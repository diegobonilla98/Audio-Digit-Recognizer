from scipy.io import wavfile
from model_checkpoint import loadModel
import numpy as np
import logging
logging.getLogger('tensorflow').disabled = True

model = loadModel("model_checkpoints/modelB2")

_, samples = wavfile.read('tests/prueba6.wav')

test_audio = samples[:, 0]

if test_audio.shape[0] > 18262:
    test_audio = test_audio[:18262]
if test_audio.shape[0] < 18262:
    padding = 18262 - test_audio.shape[0]
    test_audio = np.pad(test_audio, (0, padding), 'constant')
test_audio = test_audio.reshape((1, 18262, 1))

predictions = model.predict(test_audio)
print(np.argmax(predictions))
