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


    def getBestObservationAttributes(self):
        bestObs = None
        for obs in self._savedObservations:
            if bestObs == None or obs.getProbability() > bestObs.getProbability():
                bestObs = obs
        
        if bestObs is None:
            attrVals = {}
            if len(self._lambdaYMapping) > 0:
                for idx, attr in self._lambdaYMapping.items():
                    # We invent one with all Attributes set to 0.5 (mainly for primitive capsules)
                    if attr not in attrVals:
                        attrVals[attr] = [0.5]
                    else:
                        attrVals[attr].append(0.5)

            return attrVals

        return bestObs.getOutputsList()  # Attribute -  List of Values


    def cleanupObservations(self, applySymmetries, offsetLabelX : str, offsetLabelY : str, offsetLabelRatio : str, targetLabelX : str, targetLabelY : str, targetLabelSize : str):
        if offsetLabelX is not None and offsetLabelY is not None and offsetLabelRatio is not None:
            for observation in self._observations:
                observation.offset(offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize)


        sortedObs = sorted(self._observations, reverse = True, key = (lambda x : x.getProbability()))
        
        for index, obs in enumerate(sortedObs):
            for index2 in range(index, len(sortedObs)):
                if sortedObs[index2] in self._observations:
                    if sortedObs[index2].isZeroObservation():
                        self.removeObservation(sortedObs[index2])
                    elif index2 > index and CapsuleMemory.checkSimilarObservations(obs.getOutputs(), sortedObs[index2].getOutputs()) > HyperParameters.SimilarObservationsCutOff:
                        self.removeObservation(sortedObs[index2])
                    # TODO: Remove too small detections? TEST:
                    elif sortedObs[index2].getOutput(attributeName = "Size") * sortedObs[index2].getOutput(attributeName = "Aspect-Ratio") < 0.08:
                        self.removeObservation(sortedObs[index2])


        for observation in self._observations:
            observation.cleanupSymmetries(applySymmetries)


    def removeObservation(self, observation : Observation):
        if observation in self._observations:
            self._observations.remove(observation)
            return True
        return False



    def transformDataPoint(self, initialObservation : Observation, mainSymmetry : float, symmetries : dict):
        # symmetries   # Capsule    - Symmetry
        inputs = {}    # Attribute  - List of Values
        outputs = {}   # Attribute  - List of Values

        # TODO: Do Preposition transformations
        # TODO: Do Adjective transformations
        inputs = initialObservation.getInputs()

        outputs = self._lambdaY(inputs)

        centerX = [valueList for (key, valueList) in outputs.items() if key.getName() == "Position-X"][0][0]
        centerY = [valueList for (key, valueList) in outputs.items() if key.getName() == "Position-Y"][0][0]
        centerR = [valueList for (key, valueList) in outputs.items() if key.getName() == "Rotation"][0][0]
        centerS = [valueList for (key, valueList) in outputs.items() if key.getName() == "Size"][0][0]

        deltaX = ((centerX + (random.random() - 0.5) * 2.0) % 1.0) - centerX
        deltaY = ((centerY + (random.random() - 0.5) * 2.0) % 1.0) - centerY
        deltaSize = ((centerS + (random.random() - 0.5) * 2.0) % 1.0) - centerS
        deltaRotate = ((centerR + (random.random() - 0.5) * 2.0) % mainSymmetry) - centerR

        capsIdx = {} # Capsule - Count

        # Rotation
        for observation in initialObservation.getInputObservations():
            currentCaps = observation.getCapsule()
            xAttr = currentCaps.getAttributeByName("Position-X")
            yAttr = currentCaps.getAttributeByName("Position-Y")
            rotAttr = currentCaps.getAttributeByName("Rotation")
            sizeAttr = currentCaps.getAttributeByName("Size")

            # Hacky... Wacky...
            if currentCaps in capsIdx:
                capsIdx[currentCaps] = capsIdx[currentCaps] + 1
            else:
                capsIdx[currentCaps] = 0
            
            idx = capsIdx[currentCaps]

            # Move to Origin
            inputs[xAttr][idx] = inputs[xAttr][idx] - centerX
            inputs[yAttr][idx] = inputs[yAttr][idx] - centerY

            # Apply Rotations To Coordinates
            inputs[rotAttr][idx] = (inputs[rotAttr][idx] + deltaRotate) # % symmetries[currentCaps]
            newX = inputs[xAttr][idx] * math.cos(deltaRotate * math.pi * 2.0) - inputs[yAttr][idx] * math.sin(deltaRotate * math.pi * 2.0)
            newY = inputs[xAttr][idx] * math.sin(deltaRotate * math.pi * 2.0) + inputs[yAttr][idx] * math.cos(deltaRotate * math.pi * 2.0)

            # Apply Size
            newX = newX * (1 + (deltaSize / inputs[sizeAttr][idx]))
            newY = newY * (1 + (deltaSize / inputs[sizeAttr][idx]))
            inputs[sizeAttr][idx] = (inputs[sizeAttr][idx] + deltaSize)

            # Move away back from Origin and translate
            inputs[xAttr][idx] = newX + centerX + deltaX
            inputs[yAttr][idx] = newY + centerY + deltaY

        return inputs, self._lambdaY(inputs)   # Attribute - List of Values ,  Attribute - List of Values


    def runXInferer(self, attributes : list, isTraining : bool):
        # attributes        # Values
        return self._lambdaXInferer(attributes, isTraining) # Values


    def nextBatch(self, batchSize : int, inputMap : dict, outputMap : dict):
        # inputMap  : dict   # Attribute - List of Indices
        # outputMap : dict   # Attribute - List of Indices
        yData = [[]] * batchSize
        xData = [[]] * batchSize

        # Fill Symmetries
        # TODO: This is not complete, as only one Symmetry (ie one Route) is filled
        mainSymmetries = 1.0
        symmetries = {}
        for savedObs in self._savedObservations:
            mainSymmetries = savedObs.getCapsule().getSymmetryInverse(savedObs.getInputs())
            for obs in savedObs.getInputObservations():
                symmetries[obs.getCapsule()] = obs.getCapsule().getSymmetry(obs.getOutputsList())

        if self._lambdaXInferer is not None and self._lambdaYGenerator is not None:
            # Only create Fictive Data
            for idx in range(batchSize):
                lyData = self._lambdaYGenerator()
                lxData = self._lambdaXInferer(lyData, True)
                yData[idx] = Utility.mapData(lyData, self._lambdaYMapping, outputMap)                
                xData[idx] = Utility.mapData(lxData, self._lambdaXMapping, inputMap)
        else:
            
            lenInputMap = 0
            lenOutputMap = 0
            for idxList in inputMap.values():
                lenInputMap = lenInputMap + len(idxList)
            for idxList in outputMap.values():
                lenOutputMap = lenOutputMap + len(idxList)

            # Only create True Data + Transformations
            for idx in range(batchSize):
                xData[idx] = [0.0] * lenInputMap
                yData[idx] = [0.0] * lenOutputMap

                xVals, yVals = self.transformDataPoint(self._savedObservations[self._indexInEpoch], mainSymmetries, symmetries)

                # xVals > Attribute - List of Values
                # yVals > Attribute - List of Values

                for key, idxList in inputMap.items():
                    if key in xVals:
                        for idxidx, colIdx in enumerate(idxList):
                            xData[idx][colIdx] = xVals[key][idxidx]
                for key, idxList in outputMap.items():
                    if key in yVals:
                        for idxidx, colIdx in enumerate(idxList):
                            yData[idx][colIdx] = yVals[key][idxidx]

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