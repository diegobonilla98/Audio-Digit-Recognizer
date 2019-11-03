from scipy import signal
from scipy.io import wavfile
from PIL import Image
from os import listdir
from os.path import isfile, join

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

for num, audio in enumerate(audios):
    for idx, wav in enumerate(audio):
        sample_rate, samples = wavfile.read('recordings/' + wav)
        # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        #
        # spec_img = Image.fromarray(spectrogram).convert('LA').resize((128, 128), Image.ANTIALIAS)
        # filename = "spectrograms/num" + str(num) + "_" + str(idx) + ".png"
        # spec_img.save(filename, 'PNG')
