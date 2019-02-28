from AttributePool import AttributePool
from Capsule import Capsule
from PrimitivesRenderer import Primitives
from Observation import Observation


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
                                attributes[capsule.getAttributeByName("PixelD-" + str(xx) + "-" + str(yy))] = 0.0

                        attributes[capsule.getAttributeByName("SlidingFilter-X")] = float(offsetX + filterShape[0] / 2) / float(max(width, height))
                        attributes[capsule.getAttributeByName("SlidingFilter-Y")] = float(offsetY + filterShape[1] / 2) / float(max(width, height))

                        pixelObs = Observation(None, {}, attributes)
                        capsule.addObservation(pixelObs)

        print("Forward Pass on all Primitive Capsules")
        offsetLabelX, offsetLabelY, targetLabelX, targetLabelY = self._renderer.getOffsetLabels()

        for capsule in self._primitiveCapsules:

            # TESTING:
            return capsule.forwardPass()


            capsule.forwardPass()
            capsule.offsetObservations(offsetLabelX, offsetLabelY, targetLabelX, targetLabelY)


        for layer in range(self._numSemanticLayers):
            print("Forward Pass on Layer " + str(layer) + " of Semantic Capsules")
            for capsule in self._semanticLayers[layer]:
                capsule.forwardPass()