from NeuralNet import NeuralNet

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend


class NeuralNetPhiO(NeuralNet):

    def beginTraining(self):
        self.setTrainingParameters(10000, 1000, 4, 4)

        
    def defineModel(self, inputShape : tuple, outputSize : int):
        model = Sequential()
        
        model.add(Dense(128, activation='linear', input_shape=inputShape))
 
        #model.add(Dense(1024, activation='relu', input_shape=inputShape))
        #model.add(Dense(1024, activation='relu'))
        #model.add(Dense(1024, activation='relu'))
        #model.add(Dense(1024, activation='relu'))

        model.add(Dense(outputSize, activation='linear'))

        return model
