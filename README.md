# Audio-Digit-Recognizer
Audio recognizer using Deep Learning (convNets and transfer learning)

Personal approach to an Audio Recognizer project. First I thought in using the spectrograms to get the audio info, but it turned out in a big loss value and very low accuracy. After deciding to use the raw wav data a 1D ConvNet worked faster and better than any RNN.
The working model (audio_model.py) reached a validation accuracy of over 80% and a validation loss +-0.5.
The architecture is a couple 1D convnets and transfered learning to two more dense layers. Since the data is pretty complex, the complexity of the model and the epochs are increased.
Turned out fine. Cool project.

Learning curves.
The accuracy is on the right, the loss is the one in the left.
![Accuracy of the model](/acc.png)
![Loss of the model](/loss.png)
