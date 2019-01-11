import tensorflow as tf
from pathlib import Path

class NeuralNet:

    def __init__(self, neuralNetName : str):
        self._name : str = neuralNetName

    
    def hasTraining(self):
        fpath = Path("Models/" + self._name + ".ckpt.meta")
        if fpath.is_file():
            return True
        else:
            return False