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


class ImageRegressorN(ImageSupervised):

    def __init__(self, numOutputs : int, augment=None, **kwargs):
        self._numOutputs = numOutputs
        super().__init__(augment, **kwargs)

    @property
    def loss(self):
        return regression_loss

    @property
    def metric(self):
        return MSE

    def get_n_output_node(self):
        return self._numOutputs

    def transform_y(self, y_train):
        return y_train

    def inverse_transform_y(self, output):
        return output.flatten()

    def export_autokeras_model(self, model_file_name):
        """ Creates and Exports the AutoKeras model to the given filename. """
        portable_model = PortableImageRegressor(graph=self.cnn.best_model,
                                                y_encoder=self.y_encoder,
                                                data_transformer=self.data_transformer,
                                                resize_params=self.resize_shape,
                                                path=self.path)
        pickle_to_file(portable_model, model_file_name)



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

        self.loadFromFile()


    def setModelSplit(self, modelSplit : list):
        self._modelSplit            = modelSplit
        self._numModels             = len(modelSplit) - 1
        self._nnModel               = [None] * self._numModels

    
    def hasTraining(self):
        fpath = Path("Models/" + self._name + "-M0")
        if fpath.is_file():
            return True
        else:
            return False


    def loadFromFile(self):
        if self.hasTraining():
            for index in range(self._numModels):
                self._nnModel[index] = pickle_from_file("Models/" + self._name + "-M" + str(index))


    def trainFromData(self, trainingData : CapsuleMemory, showDebugOutput : bool = False, onlyTrain : list = None):
        if threading.current_thread() == threading.main_thread():
            numTrain = 20000 #60000
            numTest = 1000 #2000 
            timeLimit = 3*60*60 #60*60*8

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

                self._nnModel[index] = ImageRegressorN(len(Y_DeltaTrain[0]), verbose=True, augment=True)

                self._nnModel[index].fit(X_train, Y_DeltaTrain, time_limit=timeLimit)
                self._nnModel[index].final_fit(X_train, Y_DeltaTrain, X_test, Y_DeltaTest, retrain=False)

                self._nnModel[index].export_autokeras_model('Models/' + self._name + "-M" + str(index))


            '''

            # TODO: Correct Shape
            X_train = X_train.reshape((numTrain, 28, 28, 3))
            X_test = X_test.reshape((numTest, 28, 28, 3))

            self._nnModel = ImageRegressorN(self._numOutputs, verbose=True, augment=True)

            self._nnModel.fit(X_train, Y_train, time_limit=timeLimit)
            self._nnModel.final_fit(X_train, Y_train, X_test, Y_test, retrain=False)

            
            self._nnModel.export_autokeras_model('Models/' + self._name)
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
            results = np.append(results, self._nnModel[index].predict(inputs))

        return Utility.mapDataOneWayRev(results, self._outputMapping)     # Attribute - Value