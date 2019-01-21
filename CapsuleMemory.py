from Attribute import Attribute
from Utility import Utility
import numpy

class CapsuleMemory:

    def __init__(self):
        self._pMapping          : dict  = dict() # Capsule   - Column
        self._xMapping          : dict  = dict() # Attribute - Column
        self._yMapping          : dict  = dict() # Attribute - Column
        self._numEntries        : int   = 0
        self._lambdaXMapping    : dict  = dict() # Column Index - Attribute
        self._lambdaYMapping    : dict  = dict() # Column Index - Attribute
        self._lambdaYGenerator          = None   #      -> Rand Y
        self._lambdaXInferer            = None   # Y    -> X
        self._lambdaYInferer            = None   # X    -> Y

        self._indexInEpoch      : int   = 0
        self._epochsCompleted   : int   = 0
        self._scrambled         : list  = None   # Column Index


    def setLambdaKnownG(self, lambdaYGenerator, lambdaXInferer, yMapping : dict, xMapping : dict):
        # xMapping  # Column Index - Attribute
        # yMapping  # Column Index - Attribute
        self._lambdaYGenerator = lambdaYGenerator
        self._lambdaXInferer = lambdaXInferer
        self._lambdaXMapping = xMapping
        self._lambdaYMapping = yMapping
        
    def setLambdaKnownGamma(self, lambdaYInferer, xMapping : dict, yMapping : dict):
        # xMapping  # Column Index - Attribute
        # yMapping  # Column Index - Attribute
        self._lambdaYInferer = lambdaYInferer
        self._lambdaXMapping = xMapping
        self._lambdaYMapping = yMapping


    def inferXAttributes(self, attributes : list):
        # Add new Attributes
        for attribute in attributes:
            if attribute not in self._xMapping:
                self._xMapping[attribute] = [0.0] * self._numEntries
        # Remove Attributes
        removeList : list = list() 
        for attribute in self._xMapping.keys():
            if attribute not in attributes:
                removeList.append(attribute)
        for attribute in removeList:
            del self._xMapping[attribute]


    def inferYAttributes(self, attributes : list):
        # Add new Attributes
        for attribute in attributes:
            if attribute not in self._yMapping:
                self._yMapping[attribute] = [0.0] * self._numEntries
        # Remove Attributes
        removeList : list = list() 
        for attribute in self._yMapping.keys():
            if attribute not in attributes:
                removeList.append(attribute)
        for attribute in removeList:
            del self._yMapping[attribute]
            
    
    def addDataPoint(self, inputAttributes : list, outputAttributes : list, inputActivations : list):
        # Prefill, in case there are attributes missing
        for column in self._xMapping.values():
            column.append(0.0)
        for column in self._yMapping.values():
            column.append(0.0)

        self._numEntries = self._numEntries + 1

        for attribute in inputAttributes:
            if attribute in self._xMapping:
                self._xMapping[attribute][-1] = attribute.getValue()
        for attribute in outputAttributes:
            if attribute in self._yMapping:
                self._yMapping[attribute][-1] = attribute.getValue()
        for capsule, prob in inputActivations:
            if capsule in self._pMapping:
                self._pMapping[capsule].append(prob)

        self._scrambled = None
        self._indexInEpoch = 0
        self._epochsCompleted = 0


    def transformDataPoint(self, currentRow : int):
        outX = {}   # Attribute  - Value
        outY = {}   # Attribute  - Value

        for key, value in self._xMapping:
            # TODO: Transform X
            # TODO: Every nth iteration use original?
            outX[key] = value[currentRow]
            outY = Utility.mapDataOneWay(self._lambdaYInferer(Utility.mapDataOneWayDict(outX, self._lambdaXMapping)), self._lambdaYMapping)

        return outX, outY 


    def nextBatch(self, batchSize : int, inputMap : dict, outputMap : dict):
        # inputMap  : dict   # Attribute - Index
        # outputMap : dict   # Attribute - Index
        yData = [[]] * batchSize
        xData = [[]] * batchSize

        if self._scrambled is None:                
            self._scrambled = numpy.arange(0, self._numEntries) 
            numpy.random.shuffle(self._scrambled)  

        if self._numEntries == 0:
            # Only create Fictive Data
            for idx in range(batchSize):
                lyData = self._lambdaYGenerator()
                lxData = self._lambdaXInferer(lyData)
                yData[idx] = Utility.mapData(lyData, self._lambdaYMapping, outputMap)
                xData[idx] = Utility.mapData(lxData, self._lambdaXMapping, inputMap)
        else:
            # Only create True Data + Transformations
            for idx in range(batchSize):
                xData[idx] = [0.0] * len(inputMap)
                yData[idx] = [0.0] * len(outputMap)

                xVals, yVals = self.transformDataPoint(self._scrambled[self._indexInEpoch])

                for key, value in inputMap:
                    if key in self._xMapping:
                        xData[idx][value] = xVals[key]
                for key, value in outputMap:
                    if key in self._yMapping:
                        yData[idx][value] = yVals[key]

                self._indexInEpoch = self._indexInEpoch + 1
                if self._indexInEpoch > self._numEntries:
                    self._indexInEpoch = 0
                    self._epochsCompleted = self._epochsCompleted + 1                
                    self._scrambled = numpy.arange(0, self._numEntries) 
                    numpy.random.shuffle(self._scrambled) 

        return (xData, yData)