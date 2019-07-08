from AttributePool import AttributePool
from Capsule import Capsule
from PrimitivesRenderer import Primitives
from Observation import Observation
from MetaLearner import MetaLearner
from Utility import Utility

import copy
import math
import numpy as np
import matplotlib.patches as patches

import time

class CapsuleNetwork:

    def __init__(self, name : str):
        self._primitiveCapsules  : list          = []               # Capsule
        self._semanticCapsules   : list          = []               # Capsule
        self._pixelCapsules      : dict          = {}               # Shape Tuple - Capsule
        self._capsulePrimitive   : dict          = {}               # Primitive Capsule - Primitive Type
        self._semanticLayers     : dict          = {}               # Layer Index - Capsule List
        self._numSemanticLayers  : int           = 0                
        self._attributePool      : AttributePool = AttributePool()
        self._renderer                           = None             # PrimitivesRenderer Instance
        self._metaLearner        : MetaLearner   = MetaLearner()
        self._capsuleCount       : int           = 0
        self._name               : str           = name


        # Adding all Meta Learner Lambdas:
        # 1. Observed Axioms have same $\Omega$ as parent
        self._metaLearner.addLambda(lambda obs, axioms : len(self.findSameParents(list(axioms.keys()))) > 0)
        # 2. Observed Axioms don't have same $\Omega$ as parent
        self._metaLearner.addLambda(lambda obs, axioms : len(self.findSameParents(list(axioms.keys()))) == 0)
        # 3. Parts are tracked from previous scenes
        self._metaLearner.addLambda(lambda obs, axioms : \
            Utility.andElements([Utility.andElements([obsItem.hasPreviousObservation() for obsItem in obsList]) for obsList in axioms.values()]))
        # 4. Parts are NOT tracked from previous scenes
        self._metaLearner.addLambda(lambda obs, axioms : \
            not Utility.andElements([Utility.andElements([obsItem.hasPreviousObservation() for obsItem in obsList]) for obsList in axioms.values()]))
        
        # 5. $\Omega: Z(\vec{\alpha}, \vec{\tilde{\alpha}})$ indicates one attribute mismatch \\ with no entry in memory $\alpha^i >\epsilon$
        # TODO: self._metaLearner.addLambda(lambda obs, axioms : self.agreementOfMostLikelyParent(axioms))
        # 6. $\Omega: Z(\vec{\alpha}, \vec{\tilde{\alpha}})$ indicates attribute mismatch \\ for (position, rotation, size) only
        # TODO:


    def getJSON(self):
        # Only needs to save semantic capsules and data
        semanticData = []
        for layerID in range(self._numSemanticLayers):
            layerCaps = []
            for caps in self._semanticLayers[layerID]:
                layerCaps.append(caps.getJSON())

            semanticData.append({"semanticCapsules" : layerCaps})

        metaLearnerData = self._metaLearner.getJSON()

        return {"semanticLayers" : semanticData, "metaLearner" : metaLearnerData}


    def putJSON(self, data):
        # Only needs to load semantic capsules and data
        semCaps = {}
        for layerData in data["semanticLayers"]:
            for capsData in layerData["semanticCapsules"]:

                capsName = capsData["name"]
                
                obsList = []
                # First Route saved/loaded independently, as it is required to create the
                # Capsule.
                for obsData in capsData["firstRouteObservations"]:
                    obsCaps = self.getCapsuleByName(obsData["name"])
                    obsRoute = obsCaps.getRouteByName(obsData["route"])
                    obsProb = obsData["probability"]

                    attrDict = {}
                    for attrData in obsData["attributes"]:
                        attrDict[obsCaps.getAttributeByName(attrData["attribute"])] = attrData["value"]

                    obsList.append(Observation(obsCaps, obsRoute, [], attrDict, obsProb))
                
                semCaps[capsName] = self.addSemanticCapsule(capsName, obsList, 0)

                # Adding remaining memory
                if "remainingMemory" in capsData:
                    semCaps[capsName].putJSONMemory(capsData["remainingMemory"], self._attributePool, lambda name : self.getCapsuleByName(name))
        
        if "metaLearner" in data:
            self._metaLearner.putJSON(data["metaLearner"])

        return semCaps  # List of Semantic Capsules



    def getName(self):
        return self._name


    def getCapsuleByName(self, name : str):
        for caps in self._primitiveCapsules:
            if caps.getName() == name:
                return caps
        for caps in self._semanticCapsules:
            if caps.getName() == name:
                return caps
        return None


    def agreementOfMostLikelyParent(self, observedAxioms : dict):
        # observedAxioms    # {Observed Axiom (capsule), List of Observations}
        potentialParents = self.findSameParents(list(observedAxioms.keys()))

        maxProbability = 0.0
        maxAgreement = {}
        maxParent = None
        for parent in potentialParents:
            currentProbability, currentAgreement = parent.getMaxAgreement(observedAxioms)
            if currentProbability > maxProbability:
                maxParent = parent
                maxProbability = currentProbability
                maxAgreement = currentAgreement

        return maxAgreement


    def findSameParents(self, capsules : list):
        # capsules  # List of Capsules
        sameParents = []

        for caps in self._semanticCapsules:
            if caps not in capsules:
                if caps.haveSameParent(capsules) is True:
                    sameParents.append(caps)
        
        return sameParents


    def getShapeByPixelCapsule(self, capsule : Capsule):
        for shape, pixelCaps in self._pixelCapsules.items():
            if capsule == pixelCaps:
                return shape
        return (0, 0)

    
    def setRenderer(self, rendererClass):
        # rendererClass     # Class PrimitivesRenderer
        self._renderer = rendererClass(self._attributePool)

    
    def getRenderer(self):
        return self._renderer


    def addPrimitiveCapsule(self, primitive : Primitives, filterShapes : list, additionalTraining : int = 0):
        # filterShapes      # List of Tuples (width, height)
        
        if self._renderer is None:
            print("No Renderer of Type PrimitivesRenderer defined")
            return
        currentCapsule = Capsule(str(primitive), self._capsuleCount)
        self._capsulePrimitive[currentCapsule] = primitive
        self._capsuleCount = self._capsuleCount + 1
        self._renderer.createAttributesForPrimitive(primitive, currentCapsule, self._attributePool)

        for filterShape in filterShapes:
            pixelCapsule = None

            if filterShape not in self._pixelCapsules:
                pixelCapsule = Capsule("PixelLayer-" + str(filterShape[0]) + "-" + str(filterShape[1]), -1)
                self._renderer.createAttributesForPixelLayer(filterShape[0], filterShape[1], pixelCapsule, self._attributePool)
                self._pixelCapsules[filterShape] = pixelCapsule
            else:
                pixelCapsule = self._pixelCapsules[filterShape]

            currentCapsule.addPrimitiveRoute(pixelCapsule, self._renderer, primitive)
        
        self._primitiveCapsules.append(currentCapsule)

        if additionalTraining > 0:
            for i in range(additionalTraining):
                print("-------------------------------- ADDITIONAL TRAINING ROUND " + str(i + 1) + " OF " + str(additionalTraining) + " --------------------------------")
                currentCapsule.continueTraining(True, [0])

        return currentCapsule


    def addSemanticCapsule(self, name : str, fromObservations : list, additionalTraining : int = 0):
        # fromObservations          # List of Observations for one Occurance
        currentCapsule = Capsule(name, self._capsuleCount)
        self._capsuleCount = self._capsuleCount + 1
        currentCapsule.addSemanticRoute(fromObservations, self._attributePool)
        
        maxLayerID = -1
        for obs in fromObservations:
            currentLayer = self.getLayerIndex(obs.getCapsule())
            if currentLayer > maxLayerID:
                maxLayerID = currentLayer

        if self._numSemanticLayers <= maxLayerID + 1:
            self._semanticLayers[self._numSemanticLayers] = []
            self._numSemanticLayers = self._numSemanticLayers + 1

        self._semanticLayers[maxLayerID + 1].append(currentCapsule)
        self._semanticCapsules.append(currentCapsule)

        if additionalTraining > 0:
            for i in range(additionalTraining):
                print("-------------------------------- ADDITIONAL TRAINING ROUND " + str(i + 1) + " OF " + str(additionalTraining) + " --------------------------------")
                currentCapsule.continueTraining(True, [0])

        return currentCapsule


    def addSemanticTraining(self, name : str, fromObservations : list, additionalTraining : int = 0):
        # TODO:
        return

    
    def addAttribute(self, name : str, fromObservations : list, additionalTraining : int = 0):
        # TODO:
        return


    def addAttributeTraining(self, name : str, fromObservations : list, additionalTraining : int = 0):
        # TODO:
        return


    def getLayerIndex(self, capsule : Capsule):
        for layerID in range(self._numSemanticLayers):
            if capsule in self._semanticLayers[layerID]:
                return layerID
        
        return -1


    def getAttributePool(self):
        return self._attributePool

    
    def clearAllObservations(self):
        for filterShape, capsule in self._pixelCapsules.items():
            capsule.clearObservations()
        for capsule in self._primitiveCapsules:
            capsule.clearObservations()
        for capsule in self._semanticCapsules:
            capsule.clearObservations()
            

    def findObservedAxioms(self, observations : dict):
        # observations      # {Capsule, List of Observations}

        observedAxioms = {}

        # Not the fastest check, but works...
        for caps, obsList in observations.items():
            for obs in obsList:
                foundParent = False
                for checkObs in [x for checkObsList in observations.values() for x in checkObsList] :
                    if checkObs.isParent(obs):
                        foundParent = True
                        break
                if foundParent is False:
                    if caps in observedAxioms:
                        observedAxioms[caps].append(obs) 
                    else:
                        observedAxioms[caps] = [obs] 

        return observedAxioms # {Observed Axioms (Capsule), List of Observations}

    
    def showInput(self, image : list, width : int, height : int, stepSize : int = 1):
        # image             # Linear List of Pixels

        startTime = time.time()
        self.clearAllObservations()

        print("Capsule Network shown an Image of size (" + str(width) + ", " + str(height) + ")")

        offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize = self._renderer.getOffsetLabels()
        
        for filterShape, capsule in self._pixelCapsules.items():
            if filterShape[0] <= width and filterShape[1] <= height:

                fsWidth = float(filterShape[0])
                fsHeight = float(filterShape[1])
                fsMaxW = float(max(width, height))

                rads = np.fromfunction(lambda xx, yy : np.sqrt((xx / fsWidth - 0.5) ** 2 + (yy / fsHeight - 0.5) ** 2), (filterShape[0], filterShape[1]), dtype=float)
                angs = np.fromfunction(lambda xx, yy : np.arctan2((xx / fsWidth - 0.5), (yy / fsHeight - 0.5)) % (math.pi * 2), (filterShape[0], filterShape[1]), dtype=float)

                for offsetX in range(0, width - filterShape[0] + 1, stepSize):
                    for offsetY in range(0, height - filterShape[1] + 1, stepSize):
                        attributes = {}
                        for xx in range(filterShape[0]):
                            for yy in range(filterShape[1]):
                                attributes[capsule.getAttributeByName("PixelC-" + str(xx) + "-" + str(yy))] = image[((yy + offsetY) * width + xx + offsetX) * 4]
                                attributes[capsule.getAttributeByName("PixelR-" + str(xx) + "-" + str(yy))] = rads[xx, yy]
                                attributes[capsule.getAttributeByName("PixelA-" + str(xx) + "-" + str(yy))] = angs[xx, yy]
                                attributes[capsule.getAttributeByName("PixelD-" + str(xx) + "-" + str(yy))] = 1.0

                        attributes[capsule.getAttributeByName(offsetLabelX)] = float(offsetX) / fsMaxW
                        attributes[capsule.getAttributeByName(offsetLabelY)] = float(offsetY) / fsMaxW
                        attributes[capsule.getAttributeByName(offsetLabelRatio)] = fsWidth / fsMaxW

                        pixelObs = Observation(capsule, None, [], attributes, 1.0)
                        capsule.addPixelObservation(pixelObs)

        passedTime = time.time() - startTime
        print("Beginning Forward Pass on all Primitive Capsules (Time Passed: " + str(passedTime) + "s)")

        allObs = {}     # Capsule - List Of Observations

        for capsule in self._primitiveCapsules:
            capsule.forwardPass()
            capsule.cleanupObservations(offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize)
            allObs[capsule] = capsule.getObservations()

        for layer in range(self._numSemanticLayers):
            passedTime = time.time() - startTime
            print("Beginning Forward Pass on Layer " + str(layer) + " of Semantic Capsules (Time Passed: " + str(passedTime) + "s)")
            for capsule in self._semanticLayers[layer]:
                capsule.forwardPass()
                capsule.cleanupObservations()
                allObs[capsule] = capsule.getObservations()

        recommendation = None
        observedAxioms = self.findObservedAxioms(allObs)
        if len(observedAxioms) > 1:
            recommendation = self._metaLearner.checkResults(allObs, observedAxioms)

        passedTime = time.time() - startTime
        print("Total Time Passed: " + str(passedTime) + "s")

        return allObs, recommendation   # Capsule - List Of Observations, Recommendation String


    def applyOracle(self, oracleDecision : int):
        # TODO: Apply oracle decisions here?
        self._metaLearner.applyOracle(oracleDecision)

        
    def generateImage(self, width : int, height : int, observations : dict, withBackground : bool = False):
        # observations          # Capsule   -   List of Observations

        semantics = {} #  Observation - List of Patches
        texts = []     #  List of (X, Y, Text)
        offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize = self._renderer.getOffsetLabels()
        

        # We make a full copy to work on
        obs = {}    # Capsule - List of Observations
        obsMap = {} # Copied Observation - Actual Observation
        for capsule, obsList in observations.items():
            obs[capsule] = []
            for observation in obsList:
                obs[capsule].append(Observation(capsule, observation.getTakenRoute(), observation.getInputObservations(), 
                                                observation.getOutputs(), observation.getProbability()))
                obsMap[obs[capsule][-1]] = observation

        # Generate Semantic Labels for Semantic Capsules
        for capsule, obsList in obs.items():
            if capsule in self._semanticCapsules:
                for observation in obsList:
                    xOffset1 = observation.getOutput(capsule.getAttributeByName(targetLabelX))
                    yOffset1 = observation.getOutput(capsule.getAttributeByName(targetLabelY))
                    
                    xOffset1 = int(xOffset1 * float(max(width, height)))
                    yOffset1 = int(yOffset1 * float(max(width, height)))

                    semantics[obsMap[observation]] = []
    
                    for inObs in observation.getInputObservations():
                        xOffset2 = inObs.getOutput(inObs.getCapsule().getAttributeByName(targetLabelX))
                        yOffset2 = inObs.getOutput(inObs.getCapsule().getAttributeByName(targetLabelY))
                        
                        xOffset2 = int(xOffset2 * float(max(width, height)))
                        yOffset2 = int(yOffset2 * float(max(width, height)))
                        
                        semantics[obsMap[observation]].append(patches.Arrow(xOffset1, yOffset1, xOffset2 - xOffset1, yOffset2 - yOffset1, linewidth = 1, edgecolor = 'r', facecolor = 'none' ))
                        
                    semantics[obsMap[observation]].append(patches.Circle((xOffset1, yOffset1), radius = 1, color = 'r'))    


        for layerIndex in range(self._numSemanticLayers - 1, -1, -1):
            # Iterate the list backwards
            for capsule in self._semanticLayers[layerIndex]:
                if capsule in obs.keys():
                    for observation in obs[capsule]:
                        # Only top level observed symbols that have not yet been generated
                        if not observation.getInputObservations():
                            newObslist = capsule.backwardPass(observation, False)
                            for obsCaps, newObs in newObslist.items():
                                if obsCaps in obs:
                                    obs[obsCaps].extend(newObs)
                                else:
                                    obs[obsCaps] = newObs

                    # Remove the parsed Observations
                    del obs[capsule]


        # Order all observations for the primitive capsules
        capsObsPairs = []
        for capsule, obsList in obs.items():
            if capsule in self._primitiveCapsules:
                for observation in obsList:
                    capsObsPairs.append((capsule, observation))
        capsObsPairs = sorted(capsObsPairs, key=lambda tup: tup[1].getProbability())


        image = np.zeros(width * height * 4)

        for capsule, observation in capsObsPairs:            
            pixelShape = self.getShapeByPixelCapsule(capsule.getPixelLayerInput())

            xOffset = observation.getOutput(capsule.getAttributeByName(targetLabelX))
            yOffset = observation.getOutput(capsule.getAttributeByName(targetLabelY))
            size = observation.getOutput(capsule.getAttributeByName(targetLabelSize))

            observation.setOutput(capsule.getAttributeByName(targetLabelX), 0.5)
            observation.setOutput(capsule.getAttributeByName(targetLabelY), 0.5)
            observation.setOutput(capsule.getAttributeByName(targetLabelSize), size * float(max(width, height)) / float(pixelShape[0]))

            obsPixelLayer = capsule.backwardPass(observation, False)

            pixelObs = list(obsPixelLayer.values())[0][0]
            pixelLay = list(obsPixelLayer.keys())[0]

            newxOffset = int(xOffset * float(max(width, height))) - int(pixelShape[0] / 2)
            newyOffset = int(yOffset * float(max(width, height))) - int(pixelShape[1] / 2)

            # We actually have far more accurate segmentation (including rotation, etc), but its hard to do nicely in matplotlib,
            # so we decided to just box it roughly.
            minX = newxOffset + pixelShape[0]
            maxX = newxOffset
            minY = newyOffset + pixelShape[1]
            maxY = newyOffset

            for xx in range(pixelShape[0]):
                for yy in range(pixelShape[1]):
                    if xx + newxOffset < width and xx + newxOffset >= 0 and yy + newyOffset < height and yy + newyOffset >= 0:
                        depth = pixelObs.getOutput(pixelLay.getAttributeByName("PixelD-" + str(xx) + "-" + str(yy)))
                        
                        if depth < 1.0:
                            minX = min(xx + newxOffset, minX)
                            maxX = max(xx + newxOffset, maxX)
                            minY = min(yy + newyOffset, minY)
                            maxY = max(yy + newyOffset, maxY)

                        if withBackground is True or depth < 1.0:
                            image[((yy + newyOffset) * width + xx + newxOffset) * 4] = pixelObs.getOutput(pixelLay.getAttributeByName("PixelC-" + str(xx) + "-" + str(yy)))

            if observation in obsMap:
                semantics[obsMap[observation]] = [patches.Rectangle((minX, minY), maxX - minX, maxY - minY, linewidth = 1, edgecolor = 'y', facecolor = 'none')]
            else:
                semantics[observation] = [patches.Rectangle((minX, minY), maxX - minX, maxY - minY, linewidth = 1, edgecolor = 'y', facecolor = 'none')]

            observation.setOutput(capsule.getAttributeByName(targetLabelX), xOffset)
            observation.setOutput(capsule.getAttributeByName(targetLabelY), yOffset)
            observation.setOutput(capsule.getAttributeByName(targetLabelSize), size)

            texts.append((minX, minY, capsule.getName()))

        return image, semantics, texts    # Linear List of Pixels


    def producePrimitiveObservations(self, observation : Observation):
        if observation.getCapsule() not in self._primitiveCapsules:
            outputObsList = []
            if not observation.getInputObservations():
                newObsDict = observation.getCapsule().backwardPass(observation, False)
                observation.clearInputObservations()
                for newObsList in newObsDict.values():
                    for newObs in newObsList:
                        outputObsList = outputObsList + self.producePrimitiveObservations(newObs)
            else:
                for newObs in observation.getInputObservations():
                    outputObsList = outputObsList + self.producePrimitiveObservations(newObs)
            return outputObsList    # List of Observations for Primitive Capsules
        else:
            return [observation]    # List of Observation for this Primitive Capsules


    def distance(self, observationA : Observation, observationB : Observation):
        distance = 100.0
        normal1 = [0.0, 0.0]
        normal2 = [0.0, 0.0]

        compareA = self.producePrimitiveObservations(observationA)
        compareB = self.producePrimitiveObservations(observationB)

        for primA in compareA:
            for primB in compareB:
                    testDistance, testNormal1, testNormal2 = self._renderer.getDistance(
                        self._capsulePrimitive[primA.getCapsule()], self._capsulePrimitive[primB.getCapsule()],
                        primA.getOutputs(), primB.getOutputs())
                    if testDistance < distance:
                        distance = testDistance
                        normal1 = testNormal1
                        normal2 = testNormal2

        return distance, normal1, normal2

    
