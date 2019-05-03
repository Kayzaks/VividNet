from NeuralNet import NeuralNet

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend


class NeuralNetG(NeuralNet):

    def beginTraining(self):
        self.setTrainingParameters(500000, 1000, 4, 12)

        
    def defineModel(self, inputShape : tuple, outputSize : int):
        model = Sequential()
        
        model.add(Dense(512, activation='tanh', input_shape=inputShape))
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(512, activation='tanh'))
        model.add(Dense(512, activation='tanh'))

        model.add(Dense(outputSize, activation='linear'))

        return model
