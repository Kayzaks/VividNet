from Attribute import Attribute
from AttributePool import AttributePool
from CapsuleMemory import CapsuleMemory
from CapsuleRoute import CapsuleRoute
from Observation import Observation

from PrimitivesRenderer import PrimitivesRenderer
from PrimitivesRenderer import Primitives

import copy

class Capsule:

    def __init__(self, name : str):
        self._name          : str           = name    # Capsule Name / Symbol
        self._attributes    : list          = list()  # Attribute
        self._routes        : list          = list()  # Route
        self._observations  : list          = list()  # Observation


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
                (lambda attributes: knownGRenderer.renderPrimitive(knownGPrimitive, attributes, width, height, isTraining=True)),
                knownGRenderer.getModelSplit(knownGPrimitive), width, height, depth)

        self._routes.append(newRoute)


    def inheritAttributes(self, fromCapsules : list):
        for route in self._routes:
            for capsule in route.getFromCapsules():
                for attribute in capsule.getAttributes():
                    # Make sure we don't have copies
                    if attribute.getType() not in [x.getType() for x in self._attributes]:
                        newAttribute = attribute.getType().createAttribute()
                        newAttribute.setInherited()
                        self._attributes.append(newAttribute)

            route.resizeInternals()


    def createAttribute(self, name : str, attributePool : AttributePool):
        newAttribute = attributePool.createAttribute(name)
        if newAttribute is not None:
            self._attributes.append(newAttribute)

        for route in self._routes:
            route.resizeInternals()
        

    def getAttributeByName(self, name : str):
        for attr in self._attributes:
            if attr.getName().lower() == name.lower():
                return attr
        
        return None


    def getAttributes(self):
        return self._attributes


    def getMappedAttributes(self, outputMap : dict):
        # outputMap     # Index - Attribute
        outputList = []
        for key, value in sorted(outputMap.items()):
            outputList.append(value.getValue())

        return outputList


    def getAttributeValue(self, name : str):
        for attr in self._attributes:
            if attr.getName().lower() == name.lower():
                return attr.getValue()
        
        return 0.0

    def setAttributeValue(self, name : str, value : float):
        for attr in self._attributes:
            if attr.getName().lower() == name.lower():
                attr.setValue(value)


    def addObservation(self, observation : Observation):
        self._observations.append(observation)


    def clearObservations(self):
        self._observations = []


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

    def getNumObservations(self):
        return len(self._observations)


    def offsetObservations(self, offsetLabelX : str, offsetLabelY : str, targetLabelX : str, targetLabelY : str):
        for observation in self._observations:
            observation.offset(offsetLabelX, offsetLabelY, targetLabelX, targetLabelY)


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
        observations = []
        for index in range(len(self._routes[0]._fromCapsules[0]._observations)):
            permutations.append([(self._routes[0]._fromCapsules[0], index)])


        for permutation in permutations:
            inputAttributes = {}
            outputAttributes = {}       # Route - {Attribute, Value}
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
                inputs = {}
                for index, capsule in enumerate(route.getFromCapsules()):
                    inputAttributes[route][capsule] = capsule.getObservationOutput(actualPermutation[index])
                    inputs.update(inputAttributes[route][capsule])  

                # Routing by Agreement
                # 1. Run gamma                      
                outputAttributes[route] = route.runGammaFunction(inputs)

                # 2. Run g

                # 3. Calculate activation probability

                # 4. repeat for all routes

            # TODO: How to identify routes with similar inputs to differentiate?
            # 5. Find most likely route

            # TODO: If above threshold, add Observation
            observations.append(Observation(self._routes[0], inputAttributes[self._routes[0]], outputAttributes[self._routes[0]]))

        # TEMP:
        return observations
