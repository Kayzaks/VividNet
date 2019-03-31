from AttributePool import AttributePool
from Capsule import Capsule
from PrimitivesRenderer import Primitives
from Observation import Observation

import copy
import math
import numpy as np
import matplotlib.patches as patches

class CapsuleNetwork:

    def __init__(self):
        self._primitiveCapsules  : list          = []               # Capsule
        self._semanticCapsules   : list          = []               # Capsule
        self._pixelCapsules      : dict          = {}               # Shape Tuple - Capsule
        self._semanticLayers     : dict          = {}               # Layer Index - Capsule List
        self._numSemanticLayers  : int           = 0                
        self._attributePool      : AttributePool = AttributePool()
        self._renderer                           = None             # PrimitivesRenderer Instance


    def getShapeByPixelCapsule(self, capsule : Capsule):
        for shape, pixelCaps in self._pixelCapsules.items():
            if capsule == pixelCaps:
                return shape
        return (0, 0)

    
    def setRenderer(self, rendererClass):
        # rendererClass     # Class PrimitivesRenderer
        self._renderer = rendererClass(self._attributePool)


    def addPrimitiveCapsule(self, primitive : Primitives, filterShapes : list):
        # filterShapes      # List of Tuples (width, height)
        
        if self._renderer is None:
            print("No Renderer of Type PrimitivesRenderer defined")
            return
        currentCapsule = Capsule(str(primitive))
        self._renderer.createAttributesForPrimitive(primitive, currentCapsule, self._attributePool)

        for filterShape in filterShapes:
            pixelCapsule = None

            if filterShape not in self._pixelCapsules:
                pixelCapsule = Capsule("PixelLayer-" + str(filterShape[0]) + "-" + str(filterShape[1]))
                self._renderer.createAttributesForPixelLayer(filterShape[0], filterShape[1], pixelCapsule, self._attributePool)
                self._pixelCapsules[filterShape] = pixelCapsule
            else:
                pixelCapsule = self._pixelCapsules[filterShape]

            currentCapsule.addPrimitiveRoute(pixelCapsule, self._renderer, primitive)
        
        self._primitiveCapsules.append(currentCapsule)
        return currentCapsule


    def addSemanticCapsule(self, name : str, fromObservations : list):
        # fromObservations          # List of Observations for one Occurance
        currentCapsule = Capsule(name)
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
        return currentCapsule


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
            
    
    def showInput(self, image : list, width : int, height : int, stepSize : int = 1):
        # image             # Linear List of Pixels
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
                                # TODO: Use mapping from Renderer
                                attributes[capsule.getAttributeByName("PixelC-" + str(xx) + "-" + str(yy))] = image[((yy + offsetY) * width + xx + offsetX) * 4]
                                attributes[capsule.getAttributeByName("PixelR-" + str(xx) + "-" + str(yy))] = rads[xx, yy]
                                attributes[capsule.getAttributeByName("PixelA-" + str(xx) + "-" + str(yy))] = angs[xx, yy]
                                attributes[capsule.getAttributeByName("PixelD-" + str(xx) + "-" + str(yy))] = 1.0

                        attributes[capsule.getAttributeByName(offsetLabelX)] = float(offsetX) / fsMaxW
                        attributes[capsule.getAttributeByName(offsetLabelY)] = float(offsetY) / fsMaxW
                        attributes[capsule.getAttributeByName(offsetLabelRatio)] = fsWidth / fsMaxW

                        pixelObs = Observation(capsule, None, [], attributes, 1.0)
                        capsule.addPixelObservation(pixelObs)

        print("Forward Pass on all Primitive Capsules")

        allObs = {}     # Capsule - List Of Observations

        for capsule in self._primitiveCapsules:
            capsule.forwardPass()
            capsule.cleanupObservations(offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize)
            allObs[capsule] = capsule.getObservations()


        for layer in range(self._numSemanticLayers):
            print("Forward Pass on Layer " + str(layer) + " of Semantic Capsules")
            for capsule in self._semanticLayers[layer]:
                capsule.forwardPass()
                capsule.cleanupObservations()
                allObs[capsule] = capsule.getObservations()

        return allObs   # Capsule - List Of Observations

        
    def generateImage(self, width : int, height : int, observations : dict, withBackground : bool = False):
        # observations          # Capsule   -   List of Observations

        semantics = []
        texts = []
        offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize = self._renderer.getOffsetLabels()
        

        # We make a full copy to work on
        obs = {}
        for capsule, obsList in observations.items():
            obs[capsule] = []
            for observation in obsList:
                # TODO: Input Observations are still the original..
                obs[capsule].append(Observation(capsule, observation.getTakenRoute(), observation.getInputObservations(), 
                                                observation.getOutputs(), observation.getProbability()))

        # Generate Semantic Labels for Semantic Capsules
        for capsule, obsList in obs.items():
            if capsule in self._semanticCapsules:
                for observation in obsList:
                    xOffset1 = observation.getOutput(capsule.getAttributeByName(targetLabelX))
                    yOffset1 = observation.getOutput(capsule.getAttributeByName(targetLabelY))
                    
                    xOffset1 = int(xOffset1 * float(max(width, height)))
                    yOffset1 = int(yOffset1 * float(max(width, height)))

                    print(capsule.getName())
                    print(len(observation.getInputObservations()))

                    for inObs in observation.getInputObservations():
                        xOffset2 = inObs.getOutput(inObs.getCapsule().getAttributeByName(targetLabelX))
                        yOffset2 = inObs.getOutput(inObs.getCapsule().getAttributeByName(targetLabelY))
                        
                        xOffset2 = int(xOffset2 * float(max(width, height)))
                        yOffset2 = int(yOffset2 * float(max(width, height)))
                        
                        semantics.append(patches.Arrow(xOffset1, yOffset1, xOffset2 - xOffset1, yOffset2 - yOffset1, linewidth = 1, edgecolor = 'r', facecolor = 'none' ))
                        
                    semantics.append(patches.Circle((xOffset1, yOffset1), radius = 1, color = 'r'))    


        for layerIndex in range(self._numSemanticLayers - 1, -1, -1):
            # Iterate the list backwards
            for capsule in self._semanticLayers[layerIndex]:
                if capsule in obs:
                    for observation in obs[capsule]:
                        # Only top level observed symbols that have not yet been generated
                        if not observation.getInputObservations():
                            newObs = capsule.backwardPass(observation, False)
                            for obsCaps, newObs in newObs.items():
                                if obsCaps in obs:
                                    obs[obsCaps].append(newObs)
                                else:
                                    obs[obsCaps] = [newObs]

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

            pixelObs = list(obsPixelLayer.values())[0]
            pixelLay = list(obsPixelLayer.keys())[0]

            xOffset = int(xOffset * float(max(width, height))) - int(pixelShape[0] / 2)
            yOffset = int(yOffset * float(max(width, height))) - int(pixelShape[1] / 2)

            # We actually have far more accurate segmentation (including rotation, etc), but its hard to do nicely in matplotlib,
            # so we decided to just box it roughly.
            minX = xOffset + pixelShape[0]
            maxX = xOffset
            minY = yOffset + pixelShape[1]
            maxY = yOffset

            for xx in range(pixelShape[0]):
                for yy in range(pixelShape[1]):
                    if xx + xOffset < width and xx + xOffset >= 0 and yy + yOffset < height and yy + yOffset >= 0:
                        depth = pixelObs.getOutput(pixelLay.getAttributeByName("PixelD-" + str(xx) + "-" + str(yy)))
                        
                        if depth < 1.0:
                            minX = min(xx + xOffset, minX)
                            maxX = max(xx + xOffset, maxX)
                            minY = min(yy + yOffset, minY)
                            maxY = max(yy + yOffset, maxY)

                        if withBackground is True or depth < 1.0:
                            image[((yy + yOffset) * width + xx + xOffset) * 4] = pixelObs.getOutput(pixelLay.getAttributeByName("PixelC-" + str(xx) + "-" + str(yy)))

            semantics.append(patches.Rectangle((minX, minY), maxX - minX, maxY - minY, linewidth = 1, edgecolor = 'y', facecolor = 'none'))
            texts.append((minX, minY, capsule.getName()))

        return image, semantics, texts    # Linear List of Pixels