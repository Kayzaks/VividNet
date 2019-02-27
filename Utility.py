
from scipy import misc
import numpy as np


class Utility:

    @staticmethod
    def mapData(values : list, originalMap : dict, newMap : dict):
        # originalMap   # Index  - Object
        # newMap        # Object - Index
        outputs = [0.0] * len(newMap)
        for idx, val in enumerate(values):
            outputs[newMap[originalMap[idx]]] = val 

        return outputs # Values

            
    @staticmethod
    def mapDataOneWay(values : list, newMap : dict):
        # newMap        # Index - Object
        outputs : dict = dict()
        for idx, val in enumerate(values):
            outputs[newMap[idx]] = val 

        return outputs  # Object - Value


            
    @staticmethod
    def mapDataOneWayRev(values : list, newMap : dict):
        # newMap        # Object - Index
        outputs : dict = dict()
        for obj, idx in newMap.items():
            outputs[obj] = values[idx]

        return outputs  # Object - Value
        

    @staticmethod
    def mapDataOneWayDict(values : dict, newMap : dict):
        # values        # Object - Value
        # newMap        # Index  - Object
        outputs = [0.0] * len(newMap)
        for idx, obj in newMap.items():
            if obj in values:
                outputs[idx] = values[obj]

        return outputs  # Values
        
        
    @staticmethod
    def mapDataOneWayDictRev(values : dict, newMap : dict):
        # values        # Object - Value
        # newMap        # Object - Index
        outputs = [0.0] * len(newMap)
        for obj, idx in newMap.items():
            if obj in values:
                outputs[idx] = values[obj]

        return outputs  # Values


    @staticmethod
    def isqrt(n : int):
        # From https://stackoverflow.com/questions/15390807/integer-square-root-in-python
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        return x
        
        
    @staticmethod
    def loadPNGGreyscale(filename : str):
        image = misc.imread(filename, "L")
        image = np.asarray(image).astype(np.float32).flatten() / 255.0

        size = Utility.isqrt(len(image))
        output = np.zeros(size * size * 4)

        for idx in range(size * size):
            output[idx * 4] = image[idx]
            output[idx * 4 + 1] = float(idx % size) / float(size)
            output[idx * 4 + 2] = float(idx // size) / float(size)
            output[idx * 4 + 3] = 0.0

        return output 

        