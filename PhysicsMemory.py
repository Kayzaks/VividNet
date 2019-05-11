
from Memory import Memory
from PrimitivesPhysics import PrimitivesPhysics

from enum import Enum


class PhysicsMemoryMode(Enum):
    PhiR = 0
    PhiO = 1


class PhysicsMemory:

    def __init__(self):
        self._syntheticPhysics   :  PrimitivesPhysics = None
        self._generatorMode      :  PhysicsMemoryMode = PhysicsMemoryMode.PhiR

    
    def setSyntheticPhysics(self, synthPhysics : PrimitivesPhysics):
        self._syntheticPhysics = synthPhysics


    def setMode(self, mode : PhysicsMemoryMode):
        self._generatorMode = mode


    def nextBatch(self, batchSize : int, inputMap : dict, outputMap : dict):
        # inputMap  : dict   # Object - List of Indices  -> Ignored for Physics
        # outputMap : dict   # Object - List of Indices  -> Ignored for Physics
        yData = [[]] * batchSize
        xData = [[]] * batchSize

        if self._generatorMode == PhysicsMemoryMode.PhiR:
            if self._syntheticPhysics is not None:
                for i in range(batchSize):
                    xVal, yVal = self._syntheticPhysics.generateRelation()
                    xData[i] = xVal
                    yData[i] = yVal
            #else:
        else:
            if self._syntheticPhysics is not None:
                for i in range(batchSize):
                    xVal, yVal = self._syntheticPhysics.generateInteraction()
                    xData[i] = xVal
                    yData[i] = yVal
            #else:



        return (xData, yData)  # List of X Values, List of Y Values