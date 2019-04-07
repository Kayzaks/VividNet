
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


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


if __name__ == '__main__':
    testUI = GraphicsUserInterface()

    capsNet = CapsuleNetwork()
    capsNet.setRenderer(TestRenderer)

    width = 28
    height = 28
    
    squareCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Square, [(width, height)])
    circleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Circle, [(width, height)])   
    triangleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Triangle, [(width, height)])   

#    attr = { circleCapsule.getAttributeByName("Position-X") : 0.5 + 0.03 * 0.7 * math.sin(0.125 * math.pi * 2.0),
#                circleCapsule.getAttributeByName("Position-Y") :       0.5 - 0.03 * 0.7 * math.cos(0.125 * math.pi * 2.0),
#                circleCapsule.getAttributeByName("Size") : 0.20 * 0.7,
#                circleCapsule.getAttributeByName("Rotation") : 0.125,
#                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
#                circleCapsule.getAttributeByName("Intensity") : 0.8,
#                circleCapsule.getAttributeByName("Strength") : 0.1 * 0.7 }
#    
#    circObs = Observation(circleCapsule, circleCapsule._routes[0], [], attr, 1.0 )
#
#    attr = { squareCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0),
#                squareCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0),
#                squareCapsule.getAttributeByName("Size") : 0.25 * 0.7,
#                squareCapsule.getAttributeByName("Rotation") : 0.125,
#                squareCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
#                squareCapsule.getAttributeByName("Intensity") : 0.6,
#                squareCapsule.getAttributeByName("Strength") : 0.1 * 0.7 }
#    
#    sqObs = Observation(squareCapsule, squareCapsule._routes[0], [], attr, 1.0 )
#            
#    attr = { triangleCapsule.getAttributeByName("Position-X") : 0.5 + 0.0 * 0.7 * math.cos(0.125 * math.pi * 2.0) + 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0),
#                triangleCapsule.getAttributeByName("Position-Y") :       0.5 + 0.0 * 0.7 * math.sin(0.125 * math.pi * 2.0) - 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0),
#                triangleCapsule.getAttributeByName("Size") : 0.15 * 0.7,
#                triangleCapsule.getAttributeByName("Rotation") : 0.125,
#                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
#                triangleCapsule.getAttributeByName("Intensity") : 0.1,
#                triangleCapsule.getAttributeByName("Strength") : 0.4 * 0.7 }
#    
#    fin1Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )
#
#    attr = { triangleCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0) - 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0),
#                triangleCapsule.getAttributeByName("Position-Y") :       0.5 - 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0) + 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0),
#                triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * 0.7,
#                triangleCapsule.getAttributeByName("Rotation") : 0.125 + 1.0 / 12.0,
#                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
#                triangleCapsule.getAttributeByName("Intensity") : 0.1,
#                triangleCapsule.getAttributeByName("Strength") : 0.4 * 0.7 }
#    
#    fin2Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )
#
#    attr = { triangleCapsule.getAttributeByName("Position-X") : 0.5 + 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0) - 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0),
#                triangleCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * 0.7 * math.sin(0.125 * math.pi * 2.0) + 0.25 * 0.7 * math.cos(0.125 * math.pi * 2.0),
#                triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * 0.7,
#                triangleCapsule.getAttributeByName("Rotation") : (0.125 + 1.0 / 12.0 + 1.0 / 6.0) % 0.33333333,
#                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
#                triangleCapsule.getAttributeByName("Intensity") : 0.1,
#                triangleCapsule.getAttributeByName("Strength") : 0.4 * 0.7 }
#    
#    fin3Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )


    #rocketCapsule = capsNet.addSemanticCapsule("RocketTest", [sqObs, fin2Obs, fin3Obs])  
    #shuttleCapsule = capsNet.addSemanticCapsule("Shuttle", [circObs, fin1Obs])  


    def addNewSemanticCapsule(name : str, observationList : list):
        print("Training new Capsule '" + name + "' from:")
        for obs in observationList:
            obs.printOutputs(True)

        newCaps = capsNet.addSemanticCapsule(name, observationList)
      
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

        # Left Fin
        obsDicts.append({ triangleCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * spsize * math.cos(rotation * math.pi * 2.0) - 0.25 * spsize * math.sin(rotation * math.pi * 2.0),
                    triangleCapsule.getAttributeByName("Position-Y") :       0.5 - 0.25 * spsize * math.sin(rotation * math.pi * 2.0) + 0.25 * spsize * math.cos(rotation * math.pi * 2.0),
                    triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * spsize,
                    triangleCapsule.getAttributeByName("Rotation") : rotation + 1.0 / 12.0,
                    triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    triangleCapsule.getAttributeByName("Intensity") : 0.1,
                    triangleCapsule.getAttributeByName("Strength") : 0.4 * spsize })

        # Right Fin
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
        
        #imageObserved = imageReal

        for capsule in allObs.keys():
            print(str(len(allObs[capsule])) + "x " + capsule.getName())
            for index, obs in enumerate(allObs[capsule]):
                print("Observation " + str(index))
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
                    
        testUI.draw(drawPixels1, drawPixels2, shape[0], shape[1], shape[0], shape[1], semantics, texts, addNewSemanticCapsule)