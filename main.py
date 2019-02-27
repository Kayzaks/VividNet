from AttributePool import AttributePool
from Capsule import Capsule
from GraphicsUserInterface import GraphicsUserInterface
from numba import cuda, float32, int32
from Utility import Utility

import numpy as np
import random
import math

from TestPrimitives import TestPrimitives
from TestPrimitives import TestRenderer
from CapsuleNetwork import CapsuleNetwork


if __name__ == '__main__':
    testUI = GraphicsUserInterface()

    capsNet = CapsuleNetwork()
    capsNet.setRenderer(TestRenderer)

    renderer = TestRenderer(capsNet.getAttributePool())

    width = 28
    height = 28


    squareCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Square, [(width, height)])
    circleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Circle, [(width, height)])


    #circleCapsule.addNewRoute([pixelCapsule], renderer, TestPrimitives.Circle)
    #circleCapsule._routes[0].retrain(specificSplit = [0, 1])

    #for i in range(1, 3):
    #    
    #    batchX = Utility.loadPNGGreyscale("Tests/CIRCLE_" + str(i) + ".png")
    #    
    #    outputs = circleCapsule._routes[0].runGammaFunction(batchX)
    #    pixels = renderer.renderPrimitive(TestPrimitives.Circle, outputs, width, height)
    #    
    #    drawPixels1 = [0.0] * (width * height * 3)
    #    drawPixels2 = [0.0] * (width * height * 3)
    #
    #    for yy in range(height):
    #        for xx in range(width):
    #            drawPixels1[(yy * width + xx) * 3] = batchX[(yy * width + xx) * 4]
    #            drawPixels1[(yy * width + xx) * 3 + 1] = batchX[(yy * width + xx) * 4]
    #            drawPixels1[(yy * width + xx) * 3 + 2] = batchX[(yy * width + xx) * 4]
    #
    #            drawPixels2[(yy * width + xx) * 3] = pixels[(yy * width + xx) * 4]
    #            drawPixels2[(yy * width + xx) * 3 + 1] = pixels[(yy * width + xx) * 4]
    #            drawPixels2[(yy * width + xx) * 3 + 2] = pixels[(yy * width + xx) * 4]
    #            
    #    testUI.drawArrayCompare("Real", "Detected", drawPixels1, drawPixels2, width, height)


    for i in range(50):
    
        batchX, batchY = squareCapsule._routes[0].gNextBatch(1)


        outputs = squareCapsule._routes[0].runGammaFunction(batchX[0])
        pixels = renderer.renderPrimitive(TestPrimitives.Square, outputs, width, height)
        
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


