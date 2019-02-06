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
        self._nnModel               = None

        self.loadFromFile()

    
    def hasTraining(self):
        fpath = Path("Models/" + self._name)
        if fpath.is_file():
            return True
        else:
            return False


    def loadFromFile(self):
        if self._nnModel is None and self.hasTraining():
            self._nnModel = pickle_from_file("Models/" + self._name)


    def trainFromData(self, trainingData : CapsuleMemory, showDebugOutput : bool = False):
        if threading.current_thread() == threading.main_thread():
            numTrain = 60000
            numTest = 2000 
            timeLimit = 60*60*8

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

            # TODO: Correct Shape
            X_train = X_train.reshape((numTrain, 28, 28, 3))
            X_test = X_test.reshape((numTest, 28, 28, 3))

            self._nnModel = ImageRegressorN(self._numOutputs, verbose=True, augment=True)

            self._nnModel.fit(X_train, Y_train, time_limit=timeLimit)
            self._nnModel.final_fit(X_train, Y_train, X_test, Y_test, retrain=False)

            
            self._nnModel.export_autokeras_model('Models/' + self._name)


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

        results = self._nnModel.predict(inputs)
        return Utility.mapDataOneWayRev(results, self._outputMapping)        # Attribute - Value