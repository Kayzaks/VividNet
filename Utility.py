
from scipy import misc
import numpy as np


class Utility:

    @staticmethod
    def mapData(values : list, originalMap : dict, newMap : dict):
        # originalMap   # Index  - Object
        # newMap        # Object - List of Indices
        lenMap = 0
        for idxList in newMap.values():
            lenMap = lenMap + len(idxList)

        outputs = [0.0] * lenMap
        for idx, val in enumerate(values):
            for newIdx in newMap[originalMap[idx]]:
                outputs[newIdx] = val 

        return outputs # Values

            
    @staticmethod
    def mapDataOneWay(values : list, newMap : dict):
        # newMap        # Index - Object
        outputs : dict = dict()
        for idx, val in enumerate(values):
            outputs[newMap[idx]] = val 

        return outputs  # Object - Value

            
    @staticmethod
    def mapDataOneWayList(values : list, newMap : dict):
        # newMap        # Index - Object
        outputs : dict = dict()
        for idx, val in enumerate(values):
            if newMap[idx] in outputs:
                outputs[newMap[idx]].append(val)
            else:
                outputs[newMap[idx]] = [val]

        return outputs  # Object - List of Values

            
    @staticmethod
    def mapDataOneWayRev(values : list, newMap : dict):
        # newMap        # Object - Index
        outputs : dict = dict()
        for obj, idx in newMap.items():
            outputs[obj] = values[idx]

        return outputs  # Object - Value

            

    @staticmethod
    def mapDataOneWayRevList(values : list, newMap : dict):
        # newMap        # Object - List of Indices
        outputs : dict = dict()
        for obj, idxList in newMap.items():
            for idx in idxList:
                if obj in outputs:
                    outputs[obj].append(values[idx])
                else:
                    outputs[obj] = [values[idx]]

        return outputs  # Object - List of Values

        

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

        return outputs  # List of Values
        

    @staticmethod
    def mapDataOneWayDictRevList(values : dict, newMap : dict):
        # values        # Object - List of Values
        # newMap        # Object - List of Indices
        lenMap = 0
        for idxList in newMap.values():
            lenMap = lenMap + len(idxList)

        outputs = [0.0] * lenMap
        for obj, idxList in newMap.items():
            for idxidx, idx in enumerate(idxList):
                if obj in values:
                    outputs[idx] = values[obj][idxidx]

        return outputs  # List of Values
        

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
    def windowFunction(x, width, falloff):
        fullSupport = width
        linearSupport = width + falloff
        if abs(x) < fullSupport:
            return 1.0
        elif abs(x) < linearSupport:
            return (1.0 - (abs(x) - fullSupport) / (linearSupport - fullSupport))
        else:
            return 0.0
            

    @staticmethod
    def loadImage(filename : str):
        image = misc.imread(filename)

        width = len(image)
        height = len(image[0])

        outImage = [0.0] * width * height * 4

        for yy in range(height):
            for xx in range(width):
                outImage[(yy * width + xx) * 4] = float(image[yy][xx][0]) / 255.0
                outImage[(yy * width + xx) * 4 + 1] = float(image[yy][xx][1]) / 255.0
                outImage[(yy * width + xx) * 4 + 2] = float(image[yy][xx][2]) / 255.0
                outImage[(yy * width + xx) * 4 + 3] = 0.0

        return outImage, width, height
        