
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
from Observation import Observation


if __name__ == '__main__':
    testUI = GraphicsUserInterface()

    capsNet = CapsuleNetwork()
    capsNet.setRenderer(TestRenderer)

    renderer = TestRenderer(capsNet.getAttributePool())

    width = 28
    height = 28


    #squareCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Square, [(width, height)])
    circleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Circle, [(width, height)])


    for i in range(3):
        
        imageReal, shape = Utility.loadPNGGreyscale("Tests/CIRCLE_" + str(i) + ".png")

        allObs = capsNet.showInput(imageReal, shape[0], shape[1], 4)
        imageObserved = capsNet.generateImage(shape[0], shape[1], allObs, True)
            
        drawPixels1 = [0.0] * (shape[0] * shape[1] * 3)
        drawPixels2 = [0.0] * (shape[0] * shape[1] * 3)
        
        for yy in range(shape[1]):
            for xx in range(shape[0]):
                drawPixels1[(yy * shape[0] + xx) * 3] = imageReal[(yy * shape[0] + xx) * 4]
                drawPixels1[(yy * shape[0] + xx) * 3 + 1] = imageReal[(yy * shape[0] + xx) * 4]
                drawPixels1[(yy * shape[0] + xx) * 3 + 2] = imageReal[(yy * shape[0] + xx) * 4]

                drawPixels2[(yy * shape[0] + xx) * 3] = imageObserved[(yy * shape[0] + xx) * 4]
                drawPixels2[(yy * shape[0] + xx) * 3 + 1] = imageObserved[(yy * shape[0] + xx) * 4]
                drawPixels2[(yy * shape[0] + xx) * 3 + 2] = imageObserved[(yy * shape[0] + xx) * 4]
                    
        testUI.drawArrayCompare("Real", "Detected", drawPixels1, drawPixels2, shape[0], shape[1], shape[0], shape[1])