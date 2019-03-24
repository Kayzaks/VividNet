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

        model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same', input_shape=inputShape))
        model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        model.add(Dense(1028, activation='relu'))
        model.add(Dense(1028, activation='relu'))
        
        model.add(Dense(128, activation='relu'))

        model.add(Dense(outputSize, activation='linear'))       


        return model
