from NeuralNet import NeuralNet

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend


class NeuralNetGamma(NeuralNet):

    def beginTraining(self):
        self.setTrainingParameters(10000, 500, 4, 3)


    def defineModel(self, inputShape : tuple, outputSize : int):
        model = Sequential()
        
        model.add(Conv2D(8, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('tanh'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('tanh'))
        model.add(Dropout(0.25))
        model.add(Dense(outputSize))
        model.add(Activation('linear'))

        return model
