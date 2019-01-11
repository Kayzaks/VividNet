from Attribute import Attribute
import numpy

class CapsuleMemory:

    def __init__(self):
        self._xMapping          : dict  = dict() # Attribute - Column
        self._yMapping          : dict  = dict() # Attribute - Column
        self._numEntries        : int   = 0
        self._lambdaXMapping    : dict  = dict() # Column Index - Attribute
        self._lambdaYMapping    : dict  = dict() # Column Index - Attribute
        self._lambdaDataRatio   : int   = 2
        self._lambdaYGenerator          = None   #      -> Rand Y
        self._lambdaXGenerator          = None   # Y    -> X

        self._indexInEpoch      : int   = 0
        self._epochsCompleted   : int   = 0
        self._scrambled         : list  = None   # Column Index


    def setLambda(self, lambdaXGenerator, lambdaYGenerator, generatorInputsLambda, xMapping : dict, yMapping : dict):
        self._lambdaXGenerator = lambdaXGenerator
        self._lambdaYGenerator = lambdaYGenerator
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
            
    
    def addDataPoint(self, attributes):
        # Prefill, in case there are attributes missing
        for column in self._xMapping.values():
            column.append(0.0)
        for column in self._yMapping.values():
            column.append(0.0)

        self._numEntries = self._numEntries + 1

        for attribute in attributes:
            if attribute in self._xMapping:
                self._xMapping[attribute][-1] = attribute.getValue()
            if attribute in self._yMapping:
                self._yMapping[attribute][-1] = attribute.getValue()

        self._scrambled = None
        self._indexInEpoch = 0
        self._epochsCompleted = 0


    def mapData(self, values : list, originalMap : dict, newMap : dict):
        outputs = [0.0] * len(newMap)
        for idx, val in enumerate(values):
            outputs[newMap[originalMap[idx]]] = val


    def nextBatch(self, batchSize : int, inputMap : dict, outputMap : dict):
        # inputMap  : dict   # Attribute - Index
        # outputMap : dict   # Attribute - Index
        yData = [[]] * batchSize
        xData = [[]] * batchSize

        if self._scrambled is None:                
            self._scrambled = numpy.arange(0, self._numEntries) 
            numpy.random.shuffle(self._scrambled)  

        for idx in range(batchSize):

            if self._numEntries == 0 or idx % self._lambdaDataRatio:
                lyData = self._lambdaYGenerator()
                lxData = self._lambdaXGenerator(lyData)
                yData[idx] = self.mapData(lyData, self._lambdaYMapping, outputMap)
                xData[idx] = self.mapData(lxData, self._lambdaXMapping, inputMap)
            else:
                xData[idx] = [0.0] * len(inputMap)
                yData[idx] = [0.0] * len(outputMap)
                for key, value in inputMap:
                    if key in self._xMapping:
                        xData[idx][value] = self._xMapping[key][self._scrambled[self._indexInEpoch]]
                for key, value in outputMap:
                    if key in self._yMapping:
                        yData[idx][value] = self._yMapping[key][self._scrambled[self._indexInEpoch]]

                self._indexInEpoch = self._indexInEpoch + 1
                if self._indexInEpoch > self._numEntries:
                    self._indexInEpoch = 0
                    self._epochsCompleted = self._epochsCompleted + 1                
                    self._scrambled = numpy.arange(0, self._numEntries) 
                    numpy.random.shuffle(self._scrambled) 