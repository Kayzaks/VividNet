
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


    squareCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Square, [(width, height)])
    circleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Circle, [(width, height)])

    #for i in range(1):
    #    squareCapsule.continueTraining(True, [0, 1])
    #    circleCapsule.continueTraining(True, [0, 1])
        
    #for i in range(5):
    #    squareCapsule.continueTraining(True, [0, 1])
    #    circleCapsule.continueTraining(True, [0])

#    for i in range(3):        
#        imageReal, shape = Utility.loadPNGGreyscale("Tests/CIRCLE_" + str(i) + ".png")
#
#        allObs = capsNet.showInput(imageReal, shape[0], shape[1], 1)
#        imageObserved, semantics, texts = capsNet.generateImage(shape[0], shape[1], allObs, False)
#            
#        drawPixels1 = [0.0] * (shape[0] * shape[1] * 3)
#        drawPixels2 = [0.0] * (shape[0] * shape[1] * 3)
#        
#        for yy in range(shape[1]):
#            for xx in range(shape[0]):
#                drawPixels1[(yy * shape[0] + xx) * 3] = imageReal[(yy * shape[0] + xx) * 4]
#                drawPixels1[(yy * shape[0] + xx) * 3 + 1] = imageReal[(yy * shape[0] + xx) * 4]
#                drawPixels1[(yy * shape[0] + xx) * 3 + 2] = imageReal[(yy * shape[0] + xx) * 4]
#
#                drawPixels2[(yy * shape[0] + xx) * 3] = imageObserved[(yy * shape[0] + xx) * 4]
#                drawPixels2[(yy * shape[0] + xx) * 3 + 1] = imageObserved[(yy * shape[0] + xx) * 4]
#                drawPixels2[(yy * shape[0] + xx) * 3 + 2] = imageObserved[(yy * shape[0] + xx) * 4]
#                    
#        testUI.drawArrayCompare("Real", "Detected", drawPixels1, drawPixels2, shape[0], shape[1], shape[0], shape[1], semantics, [], texts, [])


    allObs = {circleCapsule : [], squareCapsule : []}
    obsDicts = []
    
    shape = (56, 56)

    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") : 0.25,
                circleCapsule.getAttributeByName("Position-Y") : 0.25,
                circleCapsule.getAttributeByName("Size") : 0.25,
                circleCapsule.getAttributeByName("Rotation") : 0.3,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 0.6,
                circleCapsule.getAttributeByName("Intensity") : 0.1,
                circleCapsule.getAttributeByName("Strength") : 0.7 })

    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") : 0.75,
                circleCapsule.getAttributeByName("Position-Y") : 0.75,
                circleCapsule.getAttributeByName("Size") : 0.25,
                circleCapsule.getAttributeByName("Rotation") : 0.7,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 0.6,
                circleCapsule.getAttributeByName("Intensity") : 0.8,
                circleCapsule.getAttributeByName("Strength") : 0.7 })

    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") : 0.75,
                circleCapsule.getAttributeByName("Position-Y") : 0.25,
                circleCapsule.getAttributeByName("Size") : 0.25,
                circleCapsule.getAttributeByName("Rotation") : 0.5,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 0.6,
                circleCapsule.getAttributeByName("Intensity") : 0.4,
                circleCapsule.getAttributeByName("Strength") : 0.7 })

    for obs in obsDicts:
        allObs[circleCapsule].append(Observation(circleCapsule._routes[0], None, obs, None, 1.0 ))

    obsDicts = []

    obsDicts.append({ squareCapsule.getAttributeByName("Position-X") : 0.25,
                squareCapsule.getAttributeByName("Position-Y") : 0.75,
                squareCapsule.getAttributeByName("Size") : 0.25,
                squareCapsule.getAttributeByName("Rotation") : 0.3,
                squareCapsule.getAttributeByName("Aspect-Ratio") : 0.6,
                squareCapsule.getAttributeByName("Intensity") : 0.6,
                squareCapsule.getAttributeByName("Strength") : 0.7 })
                
    obsDicts.append({ squareCapsule.getAttributeByName("Position-X") : 0.5,
                squareCapsule.getAttributeByName("Position-Y") : 0.5,
                squareCapsule.getAttributeByName("Size") : 0.25,
                squareCapsule.getAttributeByName("Rotation") : 0.6,
                squareCapsule.getAttributeByName("Aspect-Ratio") : 0.6,
                squareCapsule.getAttributeByName("Intensity") : 0.6,
                squareCapsule.getAttributeByName("Strength") : 0.7 })

    
    for obs in obsDicts:
        allObs[squareCapsule].append(Observation(squareCapsule._routes[0], None, obs, None, 1.0 ))




    imageReal, semantics, texts = capsNet.generateImage(shape[0], shape[0], allObs, False)
    allObs = capsNet.showInput(imageReal, shape[0], shape[1], 1)
    imageObserved, semantics, texts = capsNet.generateImage(shape[0], shape[1], allObs, False)
    
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
                
    testUI.drawAll(drawPixels1, drawPixels2, shape[0], shape[1], shape[0], shape[1], semantics, texts)