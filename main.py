from Attribute import Attribute
from AttributeType import AttributeLexical
from AttributePool import AttributePool
from Capsule import Capsule
from PrimitivesRenderer import Primitives
from PrimitivesRenderer import PrimitivesRenderer
from GraphicsUserInterface import GraphicsUserInterface

import numpy as np
import random
import math


def windowFunction(x):
    fullSupport = 0.025
    linearSupport = 0.05
    try:
        if abs(x) < fullSupport:
            return 1.0
        elif abs(x) < linearSupport:
            return (1.0 - (abs(x) - fullSupport) / (linearSupport - fullSupport))
        else:
            return 0.0
    except OverflowError:
        return 0.0


def testKernel(xx : float, yy : float, width : float, height : float, attributes : list):
    relX = xx / width
    relY = yy / height
    
    intensity = math.sqrt((relX - 0.5) * (relX - 0.5) + (relY - 0.5) * (relY - 0.5)) - attributes[0] / 2.0
    intensity = windowFunction(intensity)

    # Color and Depth
    return [intensity, xx / width, yy / height, 0.0]


class TestPrimitives(Primitives):
    Circle = 0

        
class TestRenderer(PrimitivesRenderer):
    def definePrimitives(self, attributePool : AttributePool):
        # Example Attribute
        primAttributes : dict = {}     # Index - (Name, Lexical)

        # Index must match the one used in the kernel
        primAttributes[0] = ("Alpha", AttributeLexical.NonTransmit)

        self.setPrimitiveAttributes(TestPrimitives.Circle, attributePool, primAttributes)
        self.setKernels({TestPrimitives.Circle : testKernel})


    def renderInputGenerator(self, primitive : Primitives, width : int, height : int):
        # Example Input Generator
        outList = np.random.rand(len(self._attributeLayouts[primitive]))
        return outList



if __name__ == '__main__':
    testUI = GraphicsUserInterface()
    attrPool = AttributePool()

    renderer = TestRenderer(attrPool)

    width = 28
    height = 28


    circleCapsule = Capsule("Circle-Primitive")
    renderer.createAttributesForPrimitive(TestPrimitives.Circle, circleCapsule, attrPool)


    pixelCapsule = Capsule("PixelLayer-28-28")
    renderer.createAttributesForPixelLayer(width, height, pixelCapsule, attrPool)


    circleCapsule.addNewRoute([pixelCapsule], renderer, TestPrimitives.Circle)


    for i in range(50):
        batchX, batchY = circleCapsule._routes[0].gNextBatch(1)




        outputs = circleCapsule._routes[0].runGammaFunction(batchX[0])

        pixels = renderer.renderPrimitive(TestPrimitives.Circle, outputs, width, height)
        
        drawPixels1 = [0.0] * (width * height * 3)
        drawPixels2 = [0.0] * (width * height * 3)

        for yy in range(height):
            for xx in range(width):
                drawPixels1[(yy * width + xx) * 3] = batchX[0][(yy * width + xx) * 4]
                drawPixels1[(yy * width + xx) * 3 + 1] = batchX[0][(yy * width + xx) * 4]
                drawPixels1[(yy * width + xx) * 3 + 2] = batchX[0][(yy * width + xx) * 4]

                drawPixels2[(yy * width + xx) * 3] = pixels[(yy * width + xx) * 4]
                drawPixels2[(yy * width + xx) * 3 + 1] = pixels[(yy * width + xx) * 4]
                drawPixels2[(yy * width + xx) * 3 + 2] = pixels[(yy * width + xx) * 4]
                
        testUI.drawArrayCompare("Real", "Detected", drawPixels1, drawPixels2, width, height)

