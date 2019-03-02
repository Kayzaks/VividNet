from NeuralNet import NeuralNet

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend


class NeuralNetGamma(NeuralNet):

    def beginTraining(self):
        #self.setTrainingParameters(30000, 1000, 4, 30)
        self.setTrainingParameters(30000, 1000, 4, 5)


    def defineModel(self, inputShape : tuple, outputSize : int):
        model = Sequential()

        model.add(Conv2D(16, (3, 3), activation='tanh', input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))
        #model.add(Conv2D(16, (3, 3), activation='tanh'))
        #model.add(Dropout(0.25))
        model.add(Flatten())

        model.add(Dense(256, activation='tanh'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='tanh'))
        #model.add(Dropout(0.25))

        model.add(Dense(outputSize, activation='linear'))
        
#        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=inputShape))
#        #model.add(BatchNormalization())
#        model.add(MaxPooling2D(pool_size=(2, 2)))
#        model.add(Dropout(0.25))
#
#        model.add(Conv2D(64, (3, 3), activation='relu'))
#        #model.add(BatchNormalization())
#        model.add(MaxPooling2D(pool_size=(2, 2)))
#        model.add(Dropout(0.25))
#
#        model.add(Flatten())
#
#        model.add(Dense(512, activation='relu'))
#        model.add(Dropout(0.25))
#
#        model.add(Dense(256, activation='relu'))
#        model.add(Dropout(0.25))
#
#        model.add(Dense(outputSize, activation='linear'))

        return model
