
from TestPrimitives import TestPrimitives
from TestPrimitives import TestRenderer
from CapsuleNetwork import CapsuleNetwork
from Observation import Observation
from AttributePool import AttributePool
from Capsule import Capsule
from GraphicsUserInterface import GraphicsUserInterface
from numba import cuda, float32, int32
from Utility import Utility

import numpy as np
import random
import math

# We restrict our GPU to only use 50%. This has no specific 
# reason, just to allow us to work in the background without
# CUDA running out of Memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


if __name__ == '__main__':
    testUI = GraphicsUserInterface()

    capsNet = CapsuleNetwork()
    capsNet.setRenderer(TestRenderer)

    width = 28                  # Width of the Primitive Capsule Input
    height = 28                 # Height of the Primitive Capsule Input
    shape = (84, 84)            # Total Size of the Image
    
    # Set-up of Primitive Capsules
    squareCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Square, [(width, height)], 0) #, 1)
    circleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Circle, [(width, height)], 0) #, 1) 
    triangleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Triangle, [(width, height)], 0) #, 1) 


    # ----------------------------------------------------------------------- #
    # Training Data for a Spaceship
    #
    # Alternatively, we can:
    #  1. Draw an image of the Spaceship and feed it to the Capsule Network
    #  2. Identify Primitives for Rocket/Booster/Shuttle/etc... and Train
    #  3. Restart and rotate/resize/translate the learned spaceship
    #
    # We choose the following as its quicker to test, by directly 
    # writing down some Primitives

    spsize = 0.7                # Size of the Spaceship in relation to the Screen
    attr = { circleCapsule.getAttributeByName("Position-X") :    0.5 + 0.03 * spsize * math.sin(0.125 * math.pi * 2.0),
                circleCapsule.getAttributeByName("Position-Y") : 0.5 - 0.03 * spsize * math.cos(0.125 * math.pi * 2.0),
                circleCapsule.getAttributeByName("Size") : 0.20 * spsize,
                circleCapsule.getAttributeByName("Rotation") : 0.125,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.8,
                circleCapsule.getAttributeByName("Strength") : 0.1 * spsize }
    
    circObs = Observation(circleCapsule, circleCapsule._routes[0], [], attr, 1.0 )

    attr = { squareCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * spsize * math.sin(0.125 * math.pi * 2.0),
                squareCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * spsize * math.cos(0.125 * math.pi * 2.0),
                squareCapsule.getAttributeByName("Size") : 0.25 * spsize,
                squareCapsule.getAttributeByName("Rotation") : 0.125,
                squareCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                squareCapsule.getAttributeByName("Intensity") : 0.6,
                squareCapsule.getAttributeByName("Strength") : 0.1 * spsize }
    
    sqObs = Observation(squareCapsule, squareCapsule._routes[0], [], attr, 1.0 )
            
    attr = { triangleCapsule.getAttributeByName("Position-X") :    0.5 + 0.0 * spsize * math.cos(0.125 * math.pi * 2.0) + 0.25 * spsize * math.sin(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Position-Y") : 0.5 + 0.0 * spsize * math.sin(0.125 * math.pi * 2.0) - 0.25 * spsize * math.cos(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Size") : 0.15 * spsize,
                triangleCapsule.getAttributeByName("Rotation") : 0.125,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.4 * spsize }
    
    fin1Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )

    attr = { triangleCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * spsize * math.cos(0.125 * math.pi * 2.0) - 0.25 * spsize * math.sin(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Position-Y") :       0.5 - 0.25 * spsize * math.sin(0.125 * math.pi * 2.0) + 0.25 * spsize * math.cos(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * spsize,
                triangleCapsule.getAttributeByName("Rotation") : 0.125 + 1.0 / 12.0,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.4 * spsize }
    
    fin2Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )

    attr = { triangleCapsule.getAttributeByName("Position-X") : 0.5 + 0.25 * spsize * math.cos(0.125 * math.pi * 2.0) - 0.25 * spsize * math.sin(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * spsize * math.sin(0.125 * math.pi * 2.0) + 0.25 * spsize * math.cos(0.125 * math.pi * 2.0),
                triangleCapsule.getAttributeByName("Size") : 0.25 * 0.6 * spsize,
                triangleCapsule.getAttributeByName("Rotation") : (0.125 + 1.0 / 12.0 + 1.0 / 6.0) % 0.33333333,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.4 * spsize }
    
    fin3Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], attr, 1.0 )

    # Train Semantic Capsules
    rocketCapsule = capsNet.addSemanticCapsule("Rocket", [sqObs, fin2Obs, fin3Obs], 0) #, 1) 
    shuttleCapsule = capsNet.addSemanticCapsule("Shuttle", [circObs, fin1Obs], 0) #, 1)     
    # ----------------------------------------------------------------------- #


    # UI Button Function
    def addNewSemanticCapsule(name : str, observationList : list):
        print("Training new Capsule '" + name + "' from:")
        for obs in observationList:
            obs.printOutputs(True)

        newCaps = capsNet.addSemanticCapsule(name, observationList)
        


    for i in range(3):
        
        # ----------------------------------------------------------------------- #
        # Test Image Generation
        # 
        # Alternatively we can load Images
        rotation = 0.1 * float(i) 

        allObs = {circleCapsule : [], squareCapsule : [], triangleCapsule : [], rocketCapsule : [], shuttleCapsule : []}
        
        obsDicts = []
        
        obsDicts.append({ shuttleCapsule.getAttributeByName("Position-X") :  0.5 + 0.2 * spsize * math.sin(rotation * math.pi * 2.0),
                    shuttleCapsule.getAttributeByName("Position-Y") :        0.5 - 0.2 * spsize * math.cos(rotation * math.pi * 2.0),
                    shuttleCapsule.getAttributeByName("Size") : 0.1 ,
                    shuttleCapsule.getAttributeByName("Rotation") : rotation,
                    shuttleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    shuttleCapsule.getAttributeByName("Intensity") : 0.266,
                    shuttleCapsule.getAttributeByName("Strength") : 0.3 * spsize })
        
        for obs in obsDicts:
            allObs[shuttleCapsule].append(Observation(shuttleCapsule, shuttleCapsule._routes[0], [], obs, 1.0 ))

        obsDicts = []
        obsDicts.append({ rocketCapsule.getAttributeByName("Position-X") : 0.5 - 0.25 * spsize * math.sin(rotation * math.pi * 2.0),
                    rocketCapsule.getAttributeByName("Position-Y") :       0.5 + 0.25 * spsize * math.cos(rotation * math.pi * 2.0),
                    rocketCapsule.getAttributeByName("Size") : 0.128 ,
                    rocketCapsule.getAttributeByName("Rotation") : rotation,
                    rocketCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    rocketCapsule.getAttributeByName("Intensity") : 0.266,
                    rocketCapsule.getAttributeByName("Strength") : 0.3 * spsize })
        
        for obs in obsDicts:
            allObs[rocketCapsule].append(Observation(rocketCapsule, rocketCapsule._routes[0], [], obs, 1.0 ))
            
        # Draw the image, but we ignore all semantics (or Alternatively, load images from a paint program)
        imageReal, semantics, texts = capsNet.generateImage(shape[0], shape[1], allObs, False)
        # ----------------------------------------------------------------------- #


        # Feed-Forward Pass through the Network
        allObs = capsNet.showInput(imageReal, shape[0], shape[1], 1)

        # Draw from the detected data
        imageObserved, semantics, texts = capsNet.generateImage(shape[0], shape[1], allObs, False)

        # Print all Observations
        for capsule in allObs.keys():
            print(str(len(allObs[capsule])) + "x " + capsule.getName())
            for index, obs in enumerate(allObs[capsule]):
                print("Observation " + str(index))
                obs.printOutputs(False)

        # imageObserved = imageReal

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