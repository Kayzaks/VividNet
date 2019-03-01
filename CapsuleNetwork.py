from AttributePool import AttributePool
from Capsule import Capsule
from PrimitivesRenderer import Primitives
from Observation import Observation

import copy
import numpy as np

class CapsuleNetwork:

    def __init__(self):
        self._primitiveCapsules  : list          = []               # Capsule
        self._semanticCapsules   : list          = []               # Capsule
        self._pixelCapsules      : dict          = {}               # Shape Tuple - Capsule
        self._semanticLayers     : dict          = {}               # Layer Index - Capsule List
        self._numSemanticLayers  : int           = 0                
        self._attributePool      : AttributePool = AttributePool()
        self._renderer                           = None             # PrimitivesRenderer Instance

    
    def setRenderer(self, rendererClass):
        # rendererClass     # Class PrimitivesRenderer
        self._renderer = rendererClass(self._attributePool)


    def addPrimitiveCapsule(self, primitive : Primitives, filterShapes : list):
        # filterShapes      # List of Tuples (width, height)
        
        if self._renderer is None:
            print("No Renderer of Type PrimitivesRenderer defined")
            return
        print(str(primitive) + "-Primitive")
        currentCapsule = Capsule(str(primitive) + "-Primitive")
        self._renderer.createAttributesForPrimitive(primitive, currentCapsule, self._attributePool)

        for filterShape in filterShapes:
            pixelCapsule = None

            if filterShape not in self._pixelCapsules:
                pixelCapsule = Capsule("PixelLayer-" + str(filterShape[0]) + "-" + str(filterShape[1]))
                self._renderer.createAttributesForPixelLayer(filterShape[0], filterShape[1], pixelCapsule, self._attributePool)
                self._pixelCapsules[filterShape] = pixelCapsule
            else:
                pixelCapsule = self._pixelCapsules[filterShape]

            currentCapsule.addNewRoute([pixelCapsule], self._renderer, primitive)
        
        self._primitiveCapsules.append(currentCapsule)
        return currentCapsule


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

        for filterShape, capsule in self._pixelCapsules.items():
            if filterShape[0] <= width and filterShape[1] <= height:
                for offsetX in range(0, width - filterShape[0] + 1, stepSize):
                    for offsetY in range(0, height - filterShape[1] + 1, stepSize):
                        attributes = {}
                        for xx in range(filterShape[0]):
                            for yy in range(filterShape[1]):
                                # TODO: Use mapping from Renderer
                                attributes[capsule.getAttributeByName("PixelC-" + str(xx) + "-" + str(yy))] = image[((yy + offsetY) * width + xx + offsetX) * 4]
                                attributes[capsule.getAttributeByName("PixelX-" + str(xx) + "-" + str(yy))] = float(xx) / float(filterShape[0])
                                attributes[capsule.getAttributeByName("PixelY-" + str(xx) + "-" + str(yy))] = float(yy) / float(filterShape[1])
                                attributes[capsule.getAttributeByName("PixelD-" + str(xx) + "-" + str(yy))] = 1.0

                        attributes[capsule.getAttributeByName("SlidingFilter-X")] = float(offsetX) / float(max(width, height))
                        attributes[capsule.getAttributeByName("SlidingFilter-Y")] = float(offsetY) / float(max(width, height))
                        attributes[capsule.getAttributeByName("SlidingFilter-X-Ratio")] = float(filterShape[0]) / float(max(width, height))
                        attributes[capsule.getAttributeByName("SlidingFilter-Y-Ratio")] = float(filterShape[1]) / float(max(width, height))

                        pixelObs = Observation(None, {}, attributes, None, 1.0)
                        capsule.addObservation(pixelObs)

        print("Forward Pass on all Primitive Capsules")
        offsetLabelX, offsetLabelY, offsetLabelXRatio, offsetLabelYRatio, targetLabelX, targetLabelY = self._renderer.getOffsetLabels()

        allObs = {}     # Capsule - List Of Observations

        for capsule in self._primitiveCapsules:
            capsule.forwardPass()
            capsule.cleanupObservations(offsetLabelX, offsetLabelY, offsetLabelXRatio, offsetLabelYRatio, targetLabelX, targetLabelY)
            allObs[capsule] = capsule.getObservations()


        for layer in range(self._numSemanticLayers):
            print("Forward Pass on Layer " + str(layer) + " of Semantic Capsules")
            for capsule in self._semanticLayers[layer]:
                capsule.forwardPass()
                allObs[capsule] = capsule.getObservations()

        return allObs

        


    def generateImage(self, width : int, height : int, observations : dict, withBackground : bool = False):
        # observations          # Capsule   -   List of Observations
        
        obs = copy.copy(observations)

        for layerIndex in range(self._numSemanticLayers - 1, -1, -1):
            # Iterate the list backwards
            for capsule in self._semanticLayers[layerIndex]:
                if capsule in obs:
                    for observation in obs[capsule]:
                        newObs = capsule.backwardPass(observation)
                        for obsCaps, obsList in newObs:
                            if obsCaps in obs:
                                obs[obsCaps].extend(obsList)
                            else:
                                obs[obsCaps] = obsList

                    # Remove the parsed Observations
                    del obs[capsule]

        # Order all observations for the primitive capsules
        capsObsPairs = []
        for capsule, obsList in obs.items():
            for observation in obsList:
                capsObsPairs.append((capsule, observation))
        capsObsPairs = sorted(capsObsPairs, key=lambda tup: tup[1].getProbability())

        image = np.zeros(width * height * 4)

        for capsule, observation in capsObsPairs:            
            # TODO: Nicer...
            # TODO: Get FilterShape
            xOffset = int(observation.getOutput(capsule.getAttributeByName("Position-X")) * float(max(width, height))) - 14
            yOffset = int(observation.getOutput(capsule.getAttributeByName("Position-Y")) * float(max(width, height))) - 14

            observation.setOutput(capsule.getAttributeByName("Position-X"), 0.5)
            observation.setOutput(capsule.getAttributeByName("Position-Y"), 0.5)

            obsPixelLayer = capsule.backwardPass(observation, withBackground)

            pixelObs = list(obsPixelLayer.values())[0]
            pixelLay = list(obsPixelLayer.keys())[0]
            for xx in range(28):
                for yy in range(28):
                    if xx + xOffset < width and xx + xOffset >= 0 and yy + yOffset < height and yy + yOffset >= 0:
                        depth = pixelObs.getOutput(pixelLay.getAttributeByName("PixelD-" + str(xx) + "-" + str(yy)))
                        if withBackground is True or depth < 1.0:
                            image[((yy + yOffset) * width + xx + xOffset) * 4] = pixelObs.getOutput(pixelLay.getAttributeByName("PixelC-" + str(xx) + "-" + str(yy)))

        return image    # Linear List of Pixels