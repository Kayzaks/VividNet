from Attribute import Attribute
from AttributePool import AttributePool
from CapsuleMemory import CapsuleMemory
from CapsuleRoute import CapsuleRoute
from Observation import Observation

from PrimitivesRenderer import PrimitivesRenderer
from PrimitivesRenderer import Primitives

from HyperParameters import HyperParameters

import collections
import copy
import math

class Capsule:

    def __init__(self, name : str):
        self._name          : str                       = name                      # Capsule Name / Symbol
        self._attributes    : collections.OrderedDict   = collections.OrderedDict() # Attribute Name - Attribute
        self._routes        : list                      = list()                    # Route
        self._observations  : list                      = list()                    # Observation


    def getName(self):
        return self._name


    def continueTraining(self, showDebugOutput = True, specificSplit : list = None):
        for route in self._routes:
            route.retrain(showDebugOutput, specificSplit)


    def addNewRoute(self, fromCapsules : list, knownGRenderer : PrimitivesRenderer = None, 
                          knownGPrimitive : Primitives = None):
        numRoutes = len(self._routes)
        newRoute = CapsuleRoute(self, self._name + "-R-" + str(numRoutes), fromCapsules)

        # The known g Render can only be used with one pixel-layer as input
        if knownGRenderer is not None and len(fromCapsules) == 1:
            width, height, depth = knownGRenderer.inferDimensionsFromPixelLayer(fromCapsules[0])
            
            outMapIdxAttr, outMapAttrIdx = knownGRenderer.getLambdaGOutputMap(fromCapsules[0], width, height)
            inMapIdxAttr, inMapAttrIdx = knownGRenderer.getLambdaGInputMap(knownGPrimitive, self)

            newRoute.createPrimitiveRoute(inMapAttrIdx, outMapIdxAttr, outMapAttrIdx, inMapIdxAttr,
                (lambda : knownGRenderer.renderInputGenerator(knownGPrimitive, width, height)), 
                (lambda attributes, isTraining: knownGRenderer.renderPrimitive(knownGPrimitive, attributes, width, height, isTraining)),
                (lambda attributes1, attributes2: knownGRenderer.agreementFunction(fromCapsules[0], attributes1, attributes2, width, height)),
                knownGRenderer.getModelSplit(knownGPrimitive), width, height, depth)

        self._routes.append(newRoute)

    
    def getPixelLayerInput(self):
        # Not Pretty...
        for route in self._routes:
            if route.isSemantic() is False:
                return route.getFromCapsules()[0]
        return None


    def inheritAttributes(self, fromCapsules : list):
        for route in self._routes:
            for capsule in route.getFromCapsules():
                for attribute in capsule.getAttributes():
                    # Make sure we don't have copies
                    if attribute.getType() not in [x.getType() for x in self._attributes.values()]:
                        newAttribute = attribute.getType().createAttribute()
                        newAttribute.setInherited()
                        self._attributes[newAttribute.getName()] = newAttribute

            route.resizeInternals()


    def createAttribute(self, name : str, attributePool : AttributePool):
        newAttribute = attributePool.createAttribute(name)
        if newAttribute is not None:
            self._attributes[newAttribute.getName()] = newAttribute

        for route in self._routes:
            route.resizeInternals()
        

    def getAttributeByName(self, name : str):
        if name in self._attributes:
            return self._attributes[name]
        return None


    def getAttributes(self):
        return self._attributes.values()

    
    def hasAttribute(self, attribute : Attribute):
        if attribute in self._attributes.values():
            return True
        else:
            return False


    def getMappedAttributes(self, outputMap : dict):
        # outputMap     # Index - Attribute
        outputList = []
        for key, value in sorted(outputMap.items()):
            outputList.append(value.getValue())

        return outputList


    def addObservation(self, observation : Observation):
        self._observations.append(observation)


    def clearObservations(self):
        self._observations = []


    def getObservations(self):
        return self._observations


    def getObservationOutput(self, index : int):
        if index > -1 and index < len(self._observations):
            # n-th Observation
            return self._observations[index].getOutputs()
        else:
            # "Zero" Observation
            outputDict = {}
            for attribute in self._attributes:
                outputDict[attribute] = 0.0
            return outputDict


    def getObservationProbability(self, index : int):
        if index > -1 and index < len(self._observations):
            # n-th Observation
            return self._observations[index].getProbability()
        else:
            # "Zero" Observation
            return 0.0


    def getNumObservations(self):
        return len(self._observations)


    def cleanupObservations(self, offsetLabelX : str, offsetLabelY : str, offsetLabelXRatio : str, offsetLabelYRatio : str, targetLabelX : str, targetLabelY : str):
        for observation in self._observations:
            observation.offset(offsetLabelX, offsetLabelY, offsetLabelXRatio, offsetLabelYRatio, targetLabelX, targetLabelY)

        sortedObs = sorted(self._observations, reverse = True, key = (lambda x : x.getProbability()))
        
        for index, obs in enumerate(sortedObs):
            for index2 in range(index + 1, len(sortedObs)):
                if sortedObs[index2] in self._observations:
                    agreement = CapsuleRoute.semanticAgreementFunction(obs.getOutputs(), sortedObs[index2].getOutputs())
                    if self.calculateRouteProbability(agreement) > HyperParameters.SimilarObservationsCutOff:
                        self._observations.remove(sortedObs[index2])


    def forwardPass(self):
        # TODO: For each Permutation fill capsAttributeValues 
        # Calculate num Permuations using each     getNumObservations
        # Fill some Shape with permutations and iterate that
        permutations = []   # Pairs (Capsule, Observation Index)
        # TODO: Num Entries as max number of inputs of all routes?

        # FILL PERMUTATIONS USING:
        # -1 = Zero Caps
        # i = ith entry in Obs
        # TEST:
        for index in range(len(self._routes[0]._fromCapsules[0]._observations)):
            permutations.append([(self._routes[0]._fromCapsules[0], index)])


        for permutation in permutations:
            inputAttributes = {}        # Route - {Capsule, {Attribute, Value}}
            inputProbabilities = {}     # Route - {Capsule, Probability}
            outputAttributes = {}       # Route - {Attribute, Value}
            probabilities = {}          # Route - Probability
            for route in self._routes: 

                # Zero out capsules that are not part of this route
                checkOff = copy.copy(permutation)
                actualPermutation = []
                for capsule in route.getFromCapsules():
                    found = -1
                    for index, capsObs in enumerate(checkOff):
                        if capsObs[0] == capsule:
                            actualPermutation.append(capsObs[1])
                            found = index
                            break
                    if found == -1:
                        actualPermutation.append(-1)
                    else:
                        del checkOff[found]

                inputAttributes[route] = {}
                inputProbabilities[route] = {}
                inputs = {}
                for index, capsule in enumerate(route.getFromCapsules()):
                    inputAttributes[route][capsule] = capsule.getObservationOutput(actualPermutation[index])
                    inputProbabilities[route][capsule] = capsule.getObservationProbability(actualPermutation[index])
                    inputs.update(inputAttributes[route][capsule]) 

                # Routing by Agreement
                # 1. Run gamma                      
                outputAttributes[route] = route.runGammaFunction(inputs)

                # 2. Run g
                expectedInputs = route.runGFunction(outputAttributes[route], False)

                # 3. Calculate activation probability
                agreement = route.agreementFunction(inputs, expectedInputs)
                # TODO: Rest of agreement
                probabilities[route] = self.calculateRouteProbability(agreement)

                # 4. repeat for all routes

            # TODO: How to identify routes with similar inputs to differentiate?
            # 5. Find most likely route

            # TODO: If above threshold, add Observation
            if probabilities[self._routes[0]] > HyperParameters.ProbabilityCutOff:
                self._observations.append(Observation(self._routes[0], inputAttributes[self._routes[0]], outputAttributes[self._routes[0]], inputProbabilities[self._routes[0]], probabilities[self._routes[0]]))


    def backwardPass(self, observation : Observation, withBackground : bool):
        # Observation with only outputs filled
        takenRoute = observation.getTakenRoute()
        if takenRoute is None:
            # TODO: Choose a better route
            # If no route is specified, we just take the first
            takenRoute = self._routes[0]

        outputs = takenRoute.runGFunction(observation.getOutputs(), isTraining = withBackground)
        capsAttrValues = takenRoute.pairInputCapsuleAttributes(outputs)

        obsList = {}
        for capsule, attrValues in capsAttrValues.items():
            obsList[capsule] = Observation(None, None, attrValues, None, observation.getInputProbability(capsule))

        return obsList   # Capsule - Observation (with only outputs filled)


    def calculateRouteProbability(self, agreement : dict):
        # agreement         # Attribute - Value
        if len(agreement) == 0:
            return 0.0
        total = 0.0 
        for value in agreement.values():
            total = total + value
        total = total / float(len(agreement))

        # TODO: Missing input probabilities, etc..

        return total