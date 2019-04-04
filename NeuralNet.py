
import numpy as np
from time import time
import keyboard
import threading
from Utility import Utility
from pathlib import Path
from CapsuleMemory import CapsuleMemory
from HyperParameters import HyperParameters

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend
from keras.callbacks import TensorBoard

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


class NeuralNet:
    def __init__(self, inputMapping : dict, outputMapping : dict, neuralNetName : str, swapInputOutput : bool):
                
        self._name          : str   = neuralNetName
        self._inputMapping  : dict  = inputMapping                      # Attribute - List of Indices
        self._outputMapping : dict  = outputMapping                     # Attribute - List of Indices
        self._swapInOut     : bool  = swapInputOutput

        lenInputs = 0
        lenOutputs = 0
        for idxList in inputMapping.values():
            lenInputs = lenInputs + len(idxList)
        for idxList in outputMapping.values():
            lenOutputs = lenOutputs + len(idxList)

        self._inputShape    : tuple = (1, lenInputs)
        self._numOutputs    : int   = lenOutputs

        # Keras Model
        self._modelSplit            = [0, self._numOutputs]
        self._numModels             = 1
        self._nnModel               = [None] * self._numModels
        
        # Training Attributes
        self._numTrain      : int   = 1
        self._numTest       : int   = 1
        self._batchSize     : int   = 1
        self._epochs        : int   = 1

        self.loadFromFile()



    def setModelSplit(self, modelSplit : list):
        self._modelSplit            = modelSplit
        self._numModels             = len(modelSplit) - 1
        self._nnModel               = [None] * self._numModels
        
        self.loadFromFile()


    def setInputShape(self, inputShape : list):
        self._inputShape = tuple([1] + inputShape)


    def setTrainingParameters(self, numTrain : int, numTest : int, batchSize : int, epochs : int):
        self._numTrain  = numTrain
        self._numTest   = numTest
        self._batchSize = batchSize
        self._epochs    = epochs

    
    def hasTraining(self, modelIndex : int = 0):
        fpath = Path("Models/" + self._name + "-M" + str(modelIndex) + ".h5")
        if fpath.is_file():
            return True
        else:
            return False


    def loadFromFile(self, forceReload = False):    
        for index in range(self._numModels):            
            if self._nnModel[index] is None or forceReload is True:
                if self.hasTraining(index):
                    self._nnModel[index] = keras.models.load_model("Models/" + self._name + "-M" + str(index) + ".h5", custom_objects={'rmse' : rmse})


    def defineModel(self, inputShape : tuple, outputSize : int):
        return


    def beginTraining(self):
        return


    def trainFromData(self, trainingData : CapsuleMemory, showDebugOutput : bool = True, onlyTrain : list = None, retrain : bool = False):
        if threading.current_thread() == threading.main_thread():
            self.beginTraining()

            X_train = None
            Y_train = None
            X_test = None
            Y_test = None
 
            if showDebugOutput is True:
                print("Generating Training Set (Train=" + str(self._numTrain) + ", Test=" + str(self._numTest) + ")")

            if self._swapInOut is False:
                X_train, Y_train = trainingData.nextBatch(self._numTrain, self._inputMapping, self._outputMapping)
                X_test, Y_test = trainingData.nextBatch(self._numTest, self._inputMapping, self._outputMapping)
            else:
                Y_train, X_train = trainingData.nextBatch(self._numTrain, self._outputMapping, self._inputMapping)
                Y_test, X_test = trainingData.nextBatch(self._numTest, self._outputMapping, self._inputMapping)

            if showDebugOutput is True:
                print("Done Generating Training Set")

            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            Y_train = np.asarray(Y_train)
            Y_test = np.asarray(Y_test)

            trainShape = list(self._inputShape)
            trainShape[0] = self._numTrain
            
            testShape = list(self._inputShape)
            testShape[0] = self._numTest

            X_train = X_train.reshape(tuple(trainShape))
            X_test = X_test.reshape(tuple(testShape))

            numAttributes = len(Y_train[0])
            trainList = list(range(self._numModels))


            if onlyTrain is not None:
                trainList = onlyTrain


            for index in trainList:

                Y_DeltaTrain = np.delete(Y_train, np.s_[self._modelSplit[index + 1]:numAttributes], axis=1)
                Y_DeltaTest = np.delete(Y_test, np.s_[self._modelSplit[index + 1]:numAttributes], axis=1)


                if index > 0:
                    Y_DeltaTrain = np.delete(Y_DeltaTrain, np.s_[0:self._modelSplit[index]], axis=1)
                    Y_DeltaTest = np.delete(Y_DeltaTest, np.s_[0:self._modelSplit[index]], axis=1)
    

                if retrain is True or self._nnModel[index] is None:
                    self._nnModel[index] = self.defineModel(tuple(trainShape[1:]), len(Y_DeltaTrain[0]))


                opt = keras.optimizers.Adam(lr=HyperParameters.AdamLearningRate)

                self._nnModel[index].compile(loss="mean_squared_error",
                            optimizer=opt,
                            metrics=[rmse])

                tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

                self._nnModel[index].fit(X_train, Y_DeltaTrain,
                            batch_size=self._batchSize,
                            epochs=self._epochs,
                            validation_data=(X_test, Y_DeltaTest),
                            shuffle=True, 
                            verbose=1, callbacks=[tensorboard])

                self._nnModel[index].save("Models/" + self._name + "-M" + str(index) + ".h5")
                print("Saved trained model at 'Models/" + self._name + "-M" + str(index) + ".h5'")

                scores = self._nnModel[index].evaluate(X_test, Y_DeltaTest, verbose=1)
                print("Test loss:", scores[0])
                print("Test accuracy:", scores[1])





    def forwardPass(self, inputs : dict):
        # inputs        # Attribute  -  List of Values
        
        if self.hasTraining() == False:
            print("Can't perform forward pass, as Neural Net has not been trained")
            return {}
        else:
            self.loadFromFile()

        inputs = np.asarray(Utility.mapDataOneWayDictRevList(inputs, self._inputMapping))
        inputs = inputs.reshape(self._inputShape)

        results = []

        for index in range(self._numModels):
            if self._nnModel[index] is not None:
                results = np.append(results, self._nnModel[index].predict(inputs))
            else:
                results = np.append(results, np.zeros(self._modelSplit[index + 1] - self._modelSplit[index]))

        return Utility.mapDataOneWayRevList(results, self._outputMapping)    # Attribute - List of Values