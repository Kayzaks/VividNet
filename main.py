
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

    width = 28
    height = 28
    
    squareCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Square, [(width, height)])
    circleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Circle, [(width, height)])   
    triangleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Triangle, [(width, height)])   

    attr = { circleCapsule.getAttributeByName("Position-X") : 0.5 + 0.03 * 0.7 * math.sin(0.125 * math.pi * 2.0),
                circleCapsule.getAttributeByName("Position-Y") :       0.5 - 0.03 * 0.7 * math.cos(0.125 * math.pi * 2.0),
                circleCapsule.getAttributeByName("Size") : 0.20 * 0.7,
                circleCapsule.getAttributeByName("Rotation") : 0.125,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.8,
                circleCapsule.getAttributeByName("Strength") : 0.1 * 0.7 }
    
    circObs = Observation(circleCapsule, circleCapsule._routes[0], [], attr, 1.0 )

    attr = { squareCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0),
                squareCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0),
                squareCapsule.getAttributeByName("Size") : 0.25 * 0.7,
                squareCapsule.getAttributeByName("Rotation") : 0.125,
                squareCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                squareCapsule.getAttributeByName("Intensity") : 0.6,
                squareCapsule.getAttributeByName("Strength") : 0.1 * 0.7 }
    
    sqObs = Observation(squareCapsule, squareCapsule._routes[0], [], attr, 1.0 )
            
    attr = { triangleCapsule.getAttributeByName("Position-X") : 0.5 + 0.0 * 0.7 * math.cos(0.125 * math.pi * 2.0) + 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Position-Y") :       0.5 + 0.0 * 0.7 * math.sin(0.125 * math.pi * 2.0) - 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Size") : 0.15 * 0.7,
                triangleCapsule.getAttributeByName("Rotation") : 0.125,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.4 * 0.7 }
    
    fin1Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )

    attr = { triangleCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0) - 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Position-Y") :       0.5 - 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0) + 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * 0.7,
                triangleCapsule.getAttributeByName("Rotation") : 0.125 + 1.0 / 12.0,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.4 * 0.7 }
    
    fin2Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )

    attr = { triangleCapsule.getAttributeByName("Position-X") : 0.5 + 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0) - 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0) + 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * 0.7,
                triangleCapsule.getAttributeByName("Rotation") : (0.125 + 1.0 / 12.0 + 1.0 / 6.0) % 0.33333333,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.4 * 0.7 }
    
    fin3Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )


    rocketCapsule = capsNet.addSemanticCapsule("Rocket", [sqObs, fin2Obs, fin3Obs])  
    shuttleCapsule = capsNet.addSemanticCapsule("Shuttle", [circObs, fin1Obs])  




#    attr = { rocketCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * 0.7 * math.sin(rotation * math.pi * 2.0),
#                rocketCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * 0.7 * math.cos(rotation * math.pi * 2.0),
#                rocketCapsule.getAttributeByName("Size") : 0.7,
#                rocketCapsule.getAttributeByName("Rotation") : rotation,
#                rocketCapsule.getAttributeByName("Aspect-Ratio") : 0.7,
#                rocketCapsule.getAttributeByName("Intensity") : 0.35,
#                rocketCapsule.getAttributeByName("Strength") : 0.7 }
#    
#    fin3Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )
#
#    spaceshipCapsule = capsNet.addSemanticCapsule("Shuttle", [circObs, fin1Obs])  

#        
    #for i in range(1):
    #    print("---------------- ROUND " + str(i) + "---------------")
    #    squareCapsule.continueTraining(True, [0])
    #    circleCapsule.continueTraining(True, [0])
    #    triangleCapsule.continueTraining(True, [0])


    for i in range(10):
        rotation = 0.025 + 0.1 * float(i) #0.025 + 0.05 * float(i)

        allObs = {circleCapsule : [], squareCapsule : [], triangleCapsule : []}
        
        obsDicts = []
        
        #shape = (56 , 56)
        #spsize = 0.9
        shape = (56 + 28, 56 + 28)
        spsize = 0.7

        obsDicts.append({ circleCapsule.getAttributeByName("Position-X") : 0.5 + 0.03 * spsize * math.sin(rotation * math.pi * 2.0),
                    circleCapsule.getAttributeByName("Position-Y") :       0.5 - 0.03 * spsize * math.cos(rotation * math.pi * 2.0),
                    circleCapsule.getAttributeByName("Size") : 0.20 * spsize,
                    circleCapsule.getAttributeByName("Rotation") : rotation,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.8,
                    circleCapsule.getAttributeByName("Strength") : 0.1 * spsize })
        
        for obs in obsDicts:
            allObs[circleCapsule].append(Observation(circleCapsule, circleCapsule._routes[0], [], obs, 1.0 ))

        obsDicts = []

        obsDicts.append({ squareCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * spsize * math.sin(rotation * math.pi * 2.0),
                    squareCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * spsize * math.cos(rotation * math.pi * 2.0),
                    squareCapsule.getAttributeByName("Size") : 0.25 * spsize,
                    squareCapsule.getAttributeByName("Rotation") : rotation,
                    squareCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    squareCapsule.getAttributeByName("Intensity") : 0.6,
                    squareCapsule.getAttributeByName("Strength") : 0.1 * spsize })
                    
        for obs in obsDicts:
            allObs[squareCapsule].append(Observation(squareCapsule, squareCapsule._routes[0], [], obs, 1.0 ))

        obsDicts = []

        obsDicts.append({ triangleCapsule.getAttributeByName("Position-X") : 0.5 + 0.0 * spsize * math.cos(rotation * math.pi * 2.0) + 0.25 * spsize * math.sin(rotation * math.pi * 2.0),
                    triangleCapsule.getAttributeByName("Position-Y") :       0.5 + 0.0 * spsize * math.sin(rotation * math.pi * 2.0) - 0.25 * spsize * math.cos(rotation * math.pi * 2.0),
                    triangleCapsule.getAttributeByName("Size") : 0.15 * spsize,
                    triangleCapsule.getAttributeByName("Rotation") : rotation,
                    triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    triangleCapsule.getAttributeByName("Intensity") : 0.1,
                    triangleCapsule.getAttributeByName("Strength") : 0.4 * spsize })

        obsDicts.append({ triangleCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * spsize * math.cos(rotation * math.pi * 2.0) - 0.25 * spsize * math.sin(rotation * math.pi * 2.0),
                    triangleCapsule.getAttributeByName("Position-Y") :       0.5 - 0.25 * spsize * math.sin(rotation * math.pi * 2.0) + 0.25 * spsize * math.cos(rotation * math.pi * 2.0),
                    triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * spsize,
                    triangleCapsule.getAttributeByName("Rotation") : rotation + 1.0 / 12.0,
                    triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    triangleCapsule.getAttributeByName("Intensity") : 0.1,
                    triangleCapsule.getAttributeByName("Strength") : 0.4 * spsize })

        obsDicts.append({ triangleCapsule.getAttributeByName("Position-X") : 0.5 + 0.25 * spsize * math.cos(rotation * math.pi * 2.0) - 0.25 * spsize * math.sin(rotation * math.pi * 2.0),
                    triangleCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * spsize * math.sin(rotation * math.pi * 2.0) + 0.25 * spsize * math.cos(rotation * math.pi * 2.0),
                    triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * spsize,
                    triangleCapsule.getAttributeByName("Rotation") : (rotation + 1.0 / 12.0 + 1.0 / 6.0) % 0.33333333,
                    triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    triangleCapsule.getAttributeByName("Intensity") : 0.1,
                    triangleCapsule.getAttributeByName("Strength") : 0.4 * spsize })
                    
        for obs in obsDicts:
            allObs[triangleCapsule].append(Observation(triangleCapsule, triangleCapsule._routes[0], [], obs, 1.0 ))


        imageReal, semantics, texts = capsNet.generateImage(shape[0], shape[0], allObs, False)
        allObs = capsNet.showInput(imageReal, shape[0], shape[1], 1)
        imageObserved, semantics, texts = capsNet.generateImage(shape[0], shape[1], allObs, False)
        
        #imageObserved = np.zeros((shape[0] * shape[1] * 4))

        for capsule in allObs.keys():
            print(len(allObs[capsule]))
            print(capsule.getName())
            for index, obs in enumerate(allObs[capsule]):
                print("Observation " + str(index) + " ------------")
                obs.printOutputs(False)


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