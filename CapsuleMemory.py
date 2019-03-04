from Attribute import Attribute
from Utility import Utility
from Observation import Observation
from HyperParameters import HyperParameters
import numpy
import random
import math

class CapsuleMemory:

    def __init__(self):        
        self._observations      : list  = list() # Observation
        self._savedObservations : list  = list() # Observation

        self._lambdaXMapping    : dict  = dict() # Column Index - Attribute
        self._lambdaYMapping    : dict  = dict() # Column Index - Attribute
        self._lambdaYGenerator          = None   #      -> Rand Y
        self._lambdaXInferer            = None   # Y    -> X      
        self._lambdaYInferer            = None   # X    -> Y

        self._lambdaY                   = None   # X {Attribute - Value} - Y {Attribute - Value}

        self._indexInEpoch      : int   = 0
        self._epochsCompleted   : int   = 0
        self._scrambled         : list  = None   # Column Index


    def setLambdaKnownG(self, lambdaYGenerator, lambdaXInferer, xMapping : dict, yMapping : dict):
        # xMapping  # Column Index - Attribute
        # yMapping  # Column Index - Attribute
        self._lambdaYGenerator = lambdaYGenerator
        self._lambdaXInferer = lambdaXInferer
        self._lambdaXMapping = xMapping
        self._lambdaYMapping = yMapping
        
    def setLambdaKnownGamma(self, lambdaY):
        self._lambdaY = lambdaY
            
    
    def addObservations(self, observation : list):
        self._observations.extend(observation)
    
    def addSavedObservations(self, observation : list):
        self._savedObservations.extend(observation)

    def clearObservations(self):
        # TODO: Decide when to save and when not to
        self._savedObservations.extend(self._observations)
        self._observations = []

    def getObservations(self):
        return self._observations

    def getObservation(self, index : int):
        return self._observations[index]

    def getNumObservations(self):
        return len(self._observations)




    def cleanupObservations(self, offsetLabelX : str, offsetLabelY : str, offsetLabelRatio : str, targetLabelX : str, targetLabelY : str, targetLabelSize : str):
        if offsetLabelX is not None and offsetLabelY is not None and offsetLabelRatio is not None:
            for observation in self._observations:
                observation.offset(offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize)

        sortedObs = sorted(self._observations, reverse = True, key = (lambda x : x.getProbability()))
        
        for index, obs in enumerate(sortedObs):
            for index2 in range(index + 1, len(sortedObs)):
                if sortedObs[index2] in self._observations:
                    if CapsuleMemory.checkSimilarObservations(obs.getOutputs(), sortedObs[index2].getOutputs()) > HyperParameters.SimilarObservationsCutOff:
                        self.removeObservation(sortedObs[index2])


    def removeObservation(self, observation : Observation):
        if observation in self._observations:
            self._observations.remove(observation)
            return True
        return False



    def transformDataPoint(self, observation : Observation):
        inputs = {}   # Attribute  - Value
        outputs = {}   # Attribute  - Value

        # TODO: Do Preposition transformations
        # TODO: Do Adjective transformations
        inputs = observation.getInputs()
        outputs = self._lambdaY(inputs)

        centerX = [value for (key, value) in outputs.items() if key.getName() == "Position-X"][0]
        centerY = [value for (key, value) in outputs.items() if key.getName() == "Position-Y"][0]

        deltaX = (random.random() - 0.5) * 2.0
        deltaY = (random.random() - 0.5) * 2.0
        deltaRotate = (random.random() - 0.5) * 2.0
        deltaSize = (random.random() - 0.5) * 2.0

        # Rotation
        for observation in observation.getInputObservations():
            xAttr = observation.getCapsule().getAttributeByName("Position-X")
            yAttr = observation.getCapsule().getAttributeByName("Position-Y")
            rotAttr = observation.getCapsule().getAttributeByName("Rotation")
            sizeAttr = observation.getCapsule().getAttributeByName("Size")

            # Move to Origin
            inputs[xAttr] = inputs[xAttr] - centerX
            inputs[yAttr] = inputs[yAttr] - centerY

            # Do Rotations
            #inputs[rotAttr] = (inputs[rotAttr] + deltaRotate) % 1.0
            #inputs[xAttr] = inputs[xAttr] * math.cos(-inputs[rotAttr] * math.pi * 2.0) - inputs[yAttr] * math.sin(-inputs[rotAttr] * math.pi * 2.0)
            #inputs[yAttr] = inputs[xAttr] * math.sin(-inputs[rotAttr] * math.pi * 2.0) + inputs[yAttr] * math.cos(-inputs[rotAttr] * math.pi * 2.0)

            # Move away from Origin and translate
            inputs[xAttr] = inputs[xAttr] + centerX + deltaX
            inputs[yAttr] = inputs[yAttr] + centerY + deltaY

        return inputs, self._lambdaY(inputs) 


    def runXInferer(self, attributes : list, isTraining : bool):
        # attributes        # Values
        return self._lambdaXInferer(attributes, isTraining) # Values


    def nextBatch(self, batchSize : int, inputMap : dict, outputMap : dict):
        # inputMap  : dict   # Attribute - Index
        # outputMap : dict   # Attribute - Index
        yData = [[]] * batchSize
        xData = [[]] * batchSize

        if self._lambdaXInferer is not None and self._lambdaYGenerator is not None:
            # Only create Fictive Data
            for idx in range(batchSize):
                lyData = self._lambdaYGenerator()
                lxData = self._lambdaXInferer(lyData, True)
                yData[idx] = Utility.mapData(lyData, self._lambdaYMapping, outputMap)                
                xData[idx] = Utility.mapData(lxData, self._lambdaXMapping, inputMap)
        else:
            # Only create True Data + Transformations
            for idx in range(batchSize):
                xData[idx] = [0.0] * len(inputMap)
                yData[idx] = [0.0] * len(outputMap)

                xVals, yVals = self.transformDataPoint(self._savedObservations[self._indexInEpoch])

                for key, value in inputMap.items():
                    if key in xVals:
                        xData[idx][value] = xVals[key]
                for key, value in outputMap.items():
                    if key in yVals:
                        yData[idx][value] = yVals[key]

                self._indexInEpoch = self._indexInEpoch + 1
                if self._indexInEpoch >= len(self._savedObservations):
                    self._indexInEpoch = 0
                    self._epochsCompleted = self._epochsCompleted + 1     

        return (xData, yData)


        
    @staticmethod
    def checkSimilarObservations(attributes1 : dict, attributes2 : dict):
        # attributes1       # Attribute - Value
        # attributes2       # Attribute - Value
        agreement = {}
        for attribute, value in attributes1.items():
            agreement[attribute] = Utility.windowFunction(value - attributes2[attribute], 0.1, 0.1)
            
        if len(agreement) == 0:
            return 0.0

        total = 0.0 
        for value in agreement.values():
            total = total + value
        total = total / float(len(agreement))

        return total