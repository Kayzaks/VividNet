from Attribute import Attribute
from AttributePool import AttributePool
from CapsuleMemory import CapsuleMemory
from CapsuleRoute import CapsuleRoute
from Observation import Observation
from Utility import Utility

from PrimitivesRenderer import PrimitivesRenderer
from PrimitivesRenderer import Primitives

from HyperParameters import HyperParameters

import collections
import copy
import math
import itertools


class Capsule:

    def __init__(self, name : str, orderID : int):
        self._name              : str                       = name                      # Capsule Name / Symbol
        self._orderID           : int                       = orderID                   # Capsule Order ID
        self._attributes        : collections.OrderedDict   = collections.OrderedDict() # Attribute Name - Attribute
        self._routes            : list                      = list()                    # Route

        self._pixelObservations : list                      = list()                    # List of Observations


    def getName(self):
        return self._name

    
    def getOrderID(self):
        return self._orderID


    def continueTraining(self, showDebugOutput = True, specificSplit : list = None):
        for route in self._routes:
            route.retrain(showDebugOutput, specificSplit)


    def getPrimitives(self):
        # This only works, if all inputs are filled up to the primitve layer
        # TODO: Do it for other routs than route 0
        if self._routes[0].getInputCount() == 0:
            return [self]
        else:
            outList = []
            for inputCaps in self._routes[0].getFromCapsules():
                outList = outList + inputCaps.getPrimitives()
            return outList


    def addPrimitiveRoute(self, fromCapsule, knownGRenderer : PrimitivesRenderer, 
                          knownGPrimitive : Primitives):
        numRoutes = len(self._routes)
        newRoute = CapsuleRoute(self, self._name + "-R-" + str(numRoutes), [fromCapsule])

        width, height, depth = knownGRenderer.inferDimensionsFromPixelLayer(fromCapsule)
        
        outMapIdxAttr, outMapAttrIdx = knownGRenderer.getLambdaGOutputMap(fromCapsule, width, height)
        inMapIdxAttr, inMapAttrIdx = knownGRenderer.getLambdaGInputMap(knownGPrimitive, self)

        newRoute.createPrimitiveRoute(inMapAttrIdx, outMapIdxAttr, outMapAttrIdx, inMapIdxAttr,
            (lambda : knownGRenderer.renderInputGenerator(knownGPrimitive, width, height)), 
            (lambda attributes, isTraining: knownGRenderer.renderPrimitive(knownGPrimitive, attributes, width, height, isTraining)),
            (lambda attributes1, attributes2: knownGRenderer.agreementFunction(fromCapsule, attributes1, attributes2, width, height)),
            knownGRenderer.getModelSplit(knownGPrimitive), width, height, depth)

        self._routes.append(newRoute)
        

    def addSemanticRoute(self, fromObservations : list, attributePool : AttributePool):
        numRoutes = len(self._routes)
        fromCapsules = []
        for obs in fromObservations:
            fromCapsules.append(obs.getCapsule())
        newRoute = CapsuleRoute(self, self._name + "-R-" + str(numRoutes), fromCapsules)

        self._routes.append(newRoute)

        self.inheritAttributes(fromCapsules, attributePool)

        newRoute.createSemanticRoute(fromObservations)


    def haveSameParent(self, capsules : list):
        # capsules  # List of Capsules
        for route in self._routes:
            if route.haveSameParent(capsules) is True:
                return True

        return False

    
    def getPixelLayerInput(self):
        # Not Pretty...
        for route in self._routes:
            if route.isSemantic() is False:
                return route.getFromCapsules()[0]
        return None


    def inheritAttributes(self, fromCapsules : list, attributePool : AttributePool):
        for route in self._routes:
            for capsule in route.getFromCapsules():
                for attribute in capsule.getAttributes():
                    # Make sure we don't have copies
                    if attribute.isInheritable() is True and attribute.getName() not in self._attributes:
                        self.createAttribute(attribute.getName(), attributePool, True)


    def createAttribute(self, name : str, attributePool : AttributePool, isInherited : bool = False):
        newAttribute = attributePool.createAttribute(name)
        if newAttribute is not None:
            self._attributes[newAttribute.getName()] = newAttribute
            if isInherited is True:
                self._attributes[newAttribute.getName()].setInherited()
        

    def getAttributeByName(self, name : str):
        if name in self._attributes:
            return self._attributes[name]
        return None


    def getAttributes(self):
        return list(self._attributes.values())

    
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


    def addPixelObservation(self, observation : Observation):
        self._pixelObservations.append(observation)


    def addObservations(self, route : CapsuleRoute, observation : Observation):
        route.addObservations(observation)


    def clearObservations(self):
        for route in self._routes:
            route.clearObservations()
        self._pixelObservations = []


    def getObservations(self):
        outputList = []
        for route in self._routes:
            outputList.extend(route.getObservations())

        outputList.extend(self._pixelObservations)
        return outputList


    def getObservation(self, index : int):
        if index >= 0:
            currentIndex = index
            for route in self._routes:
                if index < route.getNumObservations():
                    return route.getObservation(index)
                currentIndex = currentIndex - route.getNumObservations()
            
            if index < len(self._pixelObservations):
                return self._pixelObservations[index]

        # Otherwise, Zero Observation
        zeroDict = {}
        for attribute in self._attributes.values():
            zeroDict[attribute] = 0.0
        return Observation(self, None, [], zeroDict, 0.0)


    def getNumObservations(self):
        numObs = 0
        for route in self._routes:
            numObs = numObs + route.getNumObservations()
        numObs = numObs + len(self._pixelObservations)
        return numObs


    def cleanupObservations(self, offsetLabelX : str = None, offsetLabelY : str = None, offsetLabelRatio : str = None, targetLabelX : str = None, targetLabelY : str = None, targetLabelSize : str = None):
        for route in self._routes:
            route.cleanupObservations(offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize)


    def removeObservation(self, observation : Observation):
        for route in self._routes:
            if route.removeObservation(observation) is True:
                for inputObs in observation.getInputObservations():
                    inputObs.getCapsule().removeObservation(inputObs)


    def getAllInputs(self):
        maxNumInputs = 0
        inputCapsules = {}  # Capsule - Max Number of Occurances for all routes
        for route in self._routes:
            currInputCaps = {}
            routeCaps = route.getFromCapsules()
            maxNumInputs = max(maxNumInputs, len(routeCaps))
            for capsule in routeCaps:
                if capsule in currInputCaps:
                    currInputCaps[capsule] += 1
                else:
                    currInputCaps[capsule] = 1

            for capsule, count in currInputCaps.items():
                if capsule in inputCapsules:
                    inputCapsules[capsule] = max(inputCapsules[capsule], count)
                else:
                    inputCapsules[capsule] = count

        return maxNumInputs, inputCapsules      # Max Input Number, Capsule - Number of Occurances


    def forwardPass(self):

        # Create all Input Capsule Permutations
        maxNumCaps, allInputCaps = self.getAllInputs()
        permCapsList = [(None, -1)] * (maxNumCaps - 1)
        for capsule in allInputCaps.keys():
            for index in range(capsule.getNumObservations()):
                permCapsList.append((capsule, index))

        for permutation in itertools.permutations(permCapsList, maxNumCaps):
            inputObservations = {}      # Route - Observation
            outputAttributes = {}       # Route - {Attribute, Value}
            probabilities = {}          # Route - Probability
            for route in self._routes: 
                # Zero out capsules that are not part of this route
                # TODO: The following still produces a ton of duplicates (Due to the None's, etc..)
                #       This is "okay" as they get deleted later anyways, but an elegant
                #       way is needed to reduce these to improve performance.
                actualPermutation = []
                for index, capsule in enumerate(route.getFromCapsules()):
                    if permutation[index][0] == capsule:
                        actualPermutation.append(permutation[index][1])
                    else:
                        actualPermutation.append(-1)

                inputObservations[route] = {}   # Capsule - List of Observations
                inputs = {}                     # Attribute - List of Values
                for index, capsule in enumerate(route.getFromCapsules()):
                    if capsule in inputObservations[route]:
                        inputObservations[route][capsule].append(capsule.getObservation(actualPermutation[index]))
                    else:                        
                        inputObservations[route][capsule] = [capsule.getObservation(actualPermutation[index])]

                    for attr, val in inputObservations[route][capsule][-1].getOutputs(route.isSemantic()).items():
                        if attr in inputs:
                            inputs[attr].append(val)
                        else:
                            inputs[attr] = [val]

                # Routing by Agreement
                # 1. Run gamma                      
                outputAttributes[route] = route.runGammaFunction(inputs, False)         # Attribute - List of Values

                # 2. Run g
                expectedInputs = route.runGFunction(outputAttributes[route], False)     # Attribute - List of Values

                # 3. Calculate activation probability
                agreement = route.agreementFunction(inputs, expectedInputs)
                probabilities[route] = self.calculateRouteProbability(agreement, inputObservations[route])

                # 4. repeat for all routes

            # 5. Find most likely route
            maxRouteProbability = 0.0
            maxRoute = self._routes[0]
            for route in self._routes:
                if probabilities[route] > maxRouteProbability:
                    maxRoute = route
                    maxRouteProbability = probabilities[route]

            if probabilities[maxRoute] > maxRoute.getProbabilityCutOff():
                self.addObservations(maxRoute, [Observation(self, maxRoute, list(inputObservations[maxRoute].values()), outputAttributes[maxRoute], probabilities[maxRoute])])


    def getMaxAgreement(self, observations : dict):
        # observation   # {Capsule, List of Observations}

        # Create all Input Capsule Permutations
        maxNumCaps, allInputCaps = self.getAllInputs()
        permCapsList = [(None, -1)] * (maxNumCaps - 1)
        for capsule in allInputCaps.keys():
            for checkCapsule in observations:
                for index in range(len(observations[checkCapsule])):
                    permCapsList.append((capsule, index))


        maxProbability = 0.0
        maxAgreement = {}

        for permutation in itertools.permutations(permCapsList, maxNumCaps):
            inputObservations = {}      # Route - Observation
            outputAttributes = {}       # Route - {Attribute, Value}
            probabilities = {}          # Route - Probability
            agreement = {}              # Route - Agreement
            for route in self._routes: 
                # Zero out capsules that are not part of this route
                # TODO: The following still produces a ton of duplicates (Due to the None's, etc..)
                #       This is "okay" as they get deleted later anyways, but an elegant
                #       way is needed to reduce these to improve performance.
                actualPermutation = []
                for index, capsule in enumerate(route.getFromCapsules()):
                    if permutation[index][0] == capsule:
                        actualPermutation.append(permutation[index][1])
                    else:
                        actualPermutation.append(-1)

                inputObservations[route] = {}   # Capsule - List of Observations
                inputs = {}                     # Attribute - List of Values
                for index, capsule in enumerate(route.getFromCapsules()):
                    if capsule in inputObservations[route]:
                        if actualPermutation[index] >= 0:
                            inputObservations[route][capsule].append(observations[capsule][actualPermutation[index]])
                    else:                        
                        if actualPermutation[index] >= 0:
                            inputObservations[route][capsule] = [observations[capsule][actualPermutation[index]]]

                    for attr, val in inputObservations[route][capsule][-1].getOutputs(route.isSemantic()).items():
                        if attr in inputs:
                            inputs[attr].append(val)
                        else:
                            inputs[attr] = [val]

                # Routing by Agreement
                # 1. Run gamma                      
                outputAttributes[route] = route.runGammaFunction(inputs, False)         # Attribute - List of Values

                # 2. Run g
                expectedInputs = route.runGFunction(outputAttributes[route], False)     # Attribute - List of Values

                # 3. Calculate activation probability
                agreement[route] = route.agreementFunction(inputs, expectedInputs)
                probabilities[route] = self.calculateRouteProbability(agreement[route], inputObservations[route])

                # 4. repeat for all routes

            for route in self._routes:
                if probabilities[route] > maxProbability:
                    maxProbability = probabilities[route]
                    maxAgreement = agreement[route]

        return maxProbability, maxAgreement         # Probability, {Attribute - List of Values}


    def backwardPass(self, observation : Observation, withBackground : bool):
        # Observation with only outputs filled
        takenRoute = observation.getTakenRoute()
        if takenRoute is None:
            # TODO: Choose a better route
            # If no route is specified, we just take the first
            takenRoute = self._routes[0]

        outputs = takenRoute.runGFunction(observation.getOutputsList(), isTraining = withBackground)    # Attribute - List of Values
        capsAttrValues = takenRoute.pairInputCapsuleAttributes(outputs)                                 # Capsule - {Attribute - List of Values}

        obsList = {}
        for capsule, attrValues in capsAttrValues.items():
            obsList[capsule] = []
            for index in range(len(list(attrValues.values())[0])):
                obsList[capsule].append(Observation(capsule, None, [], attrValues, observation.getInputProbability(capsule), index))
                observation.addInputObservation(obsList[capsule][-1])    

        return obsList   # Capsule - List of Observations (with only outputs filled)


    def calculateRouteProbability(self, agreement : dict, observations : dict):
        # agreement         # Attribute - List of Values
        # observations      # Capsule - List of Observations
        
        total = 0.0
        obsCount = 0

        for capsule, observationList in observations.items():
            for observation in observationList:
                perCaps = 0.0
                attrCount = 0
                for attribute, valueList in agreement.items():
                    if capsule.hasAttribute(attribute):
                        perCaps = perCaps + sum(valueList)
                        attrCount = attrCount + len(valueList)
                # TODO: Missing the Inherited and Mean probability, however, this only "improves" probability
                # perCaps = (perCaps / float(max(1, attrCount))) * Utility.windowFunction(observation.getProbability() - 1, 0.0, 1.0)
                perCaps = (perCaps / float(max(1, attrCount)))
                total = total + perCaps
                obsCount = obsCount + 1

        total = total / obsCount

        return total

    
    def applySymmetries(self, attributes : dict):
        # TODO: Choose a better route than simply the first...
        return self._routes[0].applySymmetries(attributes)


    def getSymmetry(self, attributes : dict):
        # Symmetries from Output
        # TODO: Choose a better route than simply the first...
        return self._routes[0].getSymmetry(attributes)
        

    def getSymmetryInverse(self, attributes : dict):
        # Symmetries from Input
        # TODO: Choose a better route than simply the first...
        return self._routes[0].getSymmetryInverse(attributes)


    def getPhysicalProperties(self):
        # TODO:

        DQ = [1.0, 1.0, 1.0]    # Static = 0.0, Dynamic = 1.0
        DS = [0.0, 0.0, 0.0]    # Rigid = 0.0,  Elastic/Plastic = 1.0

        if self._name == "Figure8":
            DQ = [0.0, 0.0, 1.0]
            
        return DQ, DS