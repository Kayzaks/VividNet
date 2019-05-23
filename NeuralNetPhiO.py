from NeuralNet import NeuralNet

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend


class NeuralNetPhiO(NeuralNet):

    def beginTraining(self):
        self.setTrainingParameters(100000, 1000, 128, 40)

        
    def defineModel(self, inputShape : tuple, outputSize : int):
        model = Sequential()
        
        model.add(Dense(256, activation='relu', input_shape=inputShape))
        model.add(Dense(outputSize * 2, activation='relu'))
        
        #model.add(Dense(256, activation='linear', input_shape=inputShape))
        #model.add(Dense(outputSize * 2, activation='linear'))
        
        model.add(Dense(outputSize, activation='linear'))

        return model
