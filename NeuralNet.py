import autokeras as ak
import numpy as np
from autokeras.nn.loss_function import regression_loss
from autokeras.image.image_supervised import ImageSupervised, PortableImageRegressor
from autokeras.nn.metric import Accuracy, MSE
from autokeras.utils import pickle_to_file, pickle_from_file
import keyboard
import threading
from Utility import Utility
from pathlib import Path
from CapsuleMemory import CapsuleMemory

import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


class NeuralNet:
    def __init__(self, inputMapping : dict, outputMapping : dict, neuralNetName : str, swapInputOutput : bool):
        self._name          : str   = neuralNetName
        self._inputMapping  : dict  = inputMapping                      # Attribute - Index
        self._outputMapping : dict  = outputMapping                     # Attribute - Index
        self._numInputs     : int   = max(inputMapping.values())  + 1
        self._numOutputs    : int   = max(outputMapping.values()) + 1
        self._swapInOut     : bool  = swapInputOutput

        # Auto-Keras Model
        self._modelSplit            = [0, self._numOutputs]
        self._numModels             = 1
        self._nnModel               = [None] * self._numModels

        
        #backend.set_session(backend.tf.Session(config=backend.tf.ConfigProto( 
        #    intra_op_parallelism_threads=2, 
        #    inter_op_parallelism_threads=2,
        #    allow_soft_placement=True,
        #    device_count = {'CPU': 2})))

        self.loadFromFile()


    def setModelSplit(self, modelSplit : list):
        self._modelSplit            = modelSplit
        self._numModels             = len(modelSplit) - 1
        self._nnModel               = [None] * self._numModels

    
    def hasTraining(self, modelIndex : int = 0):
        fpath = Path("Models/" + self._name + "-M" + str(modelIndex) + ".h5")
        if fpath.is_file():
            return True
        else:
            return False


    def loadFromFile(self):    
        for index in range(self._numModels):            
            if self.hasTraining(index):
                self._nnModel[index] = keras.models.load_model("Models/" + self._name + "-M" + str(index) + ".h5", custom_objects={'rmse' : rmse})


    def trainFromData(self, trainingData : CapsuleMemory, showDebugOutput : bool = False, onlyTrain : list = None):
        if threading.current_thread() == threading.main_thread():
            numTrain = 10000 #20000 # 60000
            numTest = 500 #2000 
            batch_size = 4
            epochs = 10
            #timeLimit = 3*60*60 #4*60*60 #60*60*8

            X_train = None
            Y_train = None
            X_test = None
            Y_test = None
 
            if self._swapInOut is False:
                X_train, Y_train = trainingData.nextBatch(numTrain, self._inputMapping, self._outputMapping)
                X_test, Y_test = trainingData.nextBatch(numTest, self._inputMapping, self._outputMapping)
            else:
                Y_train, X_train = trainingData.nextBatch(numTrain, self._outputMapping, self._inputMapping)
                Y_test, X_test = trainingData.nextBatch(numTest, self._outputMapping, self._inputMapping)


            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            Y_train = np.asarray(Y_train)
            Y_test = np.asarray(Y_test)

            
            # TODO: Correct Shape, Width and Height
            X_train = X_train.reshape((numTrain, 28, 28, 3))
            X_test = X_test.reshape((numTest, 28, 28, 3))

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
    
                self._nnModel[index] = Sequential()
                
                self._nnModel[index].add(Conv2D(8, (3, 3), padding='same', input_shape=(28, 28, 3)))
                self._nnModel[index].add(Activation('tanh'))
                self._nnModel[index].add(Dropout(0.25))

                self._nnModel[index].add(Flatten())
                self._nnModel[index].add(Dense(128))
                self._nnModel[index].add(Activation('tanh'))
                self._nnModel[index].add(Dropout(0.25))
                self._nnModel[index].add(Dense(len(Y_DeltaTrain[0])))
                self._nnModel[index].add(Activation('linear'))




                ''' BEST:
                
                learn rate = 0.0001
                loss = 0.0485
                val loss = 0.0439

                numTrain = 10000 
                numTest = 500 
                batch_size = 4
                epochs = 10
                
                self._nnModel[index].add(Conv2D(8, (3, 3), padding='same', input_shape=(28, 28, 3)))
                self._nnModel[index].add(Activation('tanh'))
                self._nnModel[index].add(Dropout(0.25))

                self._nnModel[index].add(Flatten())
                self._nnModel[index].add(Dense(128))
                self._nnModel[index].add(Activation('tanh'))
                self._nnModel[index].add(Dropout(0.25))
                self._nnModel[index].add(Dense(numAttributes))
                self._nnModel[index].add(Activation('linear'))
                '''



                opt = keras.optimizers.Adam(lr=0.0001)


                self._nnModel[index].compile(loss='mean_squared_error',
                            optimizer=opt,
                            metrics=[rmse])

                self._nnModel[index].fit(X_train, Y_DeltaTrain,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(X_test, Y_DeltaTest),
                            shuffle=True)


                self._nnModel[index].save("Models/" + self._name + "-M" + str(index) + ".h5")
                print("Saved trained model at 'Models/" + self._name + "-M" + str(index) + ".h5'")

                # Score trained model.
                scores = self._nnModel[index].evaluate(X_test, Y_DeltaTest, verbose=1)
                print('Test loss:', scores[0])
                print('Test accuracy:', scores[1])

                '''
                self._nnModel[index] = ImageRegressorN(len(Y_DeltaTrain[0]), verbose=True, augment=True)

                self._nnModel[index].fit(X_train, Y_DeltaTrain, time_limit=timeLimit)
                self._nnModel[index].final_fit(X_train, Y_DeltaTrain, X_test, Y_DeltaTest, retrain=False)

                self._nnModel[index].export_autokeras_model('Models/' + self._name + "-M" + str(index))
                '''



    def forwardPass(self, inputs : dict):
        # inputs        # Attribute  -  Value
        
        if self.hasTraining() == False:
            print("Can't perform forward pass, as Neural Net has not been trained")
            return {}
        else:
            self.loadFromFile()

        # TODO: Correct Shape
        inputs = np.asarray(Utility.mapDataOneWayDictRev(inputs, self._inputMapping))
        inputs = inputs.reshape((1, 28, 28, 3))

        results = []

        for index in range(self._numModels):
            if self._nnModel[index] is not None:
                results = np.append(results, self._nnModel[index].predict(inputs))
            else:
                results = np.append(results, np.zeros(self._modelSplit[index + 1] - self._modelSplit[index]))

        return Utility.mapDataOneWayRev(results, self._outputMapping)    # Attribute - Value