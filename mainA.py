
# ------------------------------------------------------------------
#
# Note1: This is highly un-optimized. It may take several minutes to 
#        process a single image. The constant context switching 
#        between rendering and neural networks seems to be the main
#        culprit.
#
# Note2: Due to size constraints, we had to omit the trained models
#        for the primitive capsules. Running this file will automatically
#        re-train them, however, this might take over 4 hours (as tested
#        on a Laptop with a nVidia GTX 1060)


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

#  We restrict our GPU to only use X %. This has no specific 
#  reason, just to allow us to work in the background without
#  CUDA running out of Memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


if __name__ == '__main__':
    testUI = GraphicsUserInterface()

    capsNet = CapsuleNetwork()
    capsNet.setRenderer(TestRenderer)

    primWidth = 28              # Width of the Primitive Capsule Input
    primHeight = 28             # Height of the Primitive Capsule Input2
    
    retrain = 1                 # After initial Training (2 loops), set this to 0 to avoid
                                # retraining each run

    # Set-up of Primitive Capsules
    squareCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Square, [(primWidth, primHeight)], retrain)
    circleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Circle, [(primWidth, primHeight)], retrain) 
    triangleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Triangle, [(primWidth, primHeight)], retrain) 


    # We dont have a file-format to save the structure of the capsule network yet, so we 
    # do it in code and set up all the routes by hand. The following is one memory entry
    # for the Capsule network, as highlighted in the paper.
    
    circObs = Observation(circleCapsule, circleCapsule._routes[0], [], 
            { circleCapsule.getAttributeByName("Position-X") : 0.5148492424049175,
                circleCapsule.getAttributeByName("Position-Y") : 0.4851507575950825,
                circleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                circleCapsule.getAttributeByName("Rotation") : 0.125,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.8,
                circleCapsule.getAttributeByName("Strength") : 0.07 }, 1.0 )
    
    sqObs = Observation(squareCapsule, squareCapsule._routes[0], [], 
            { squareCapsule.getAttributeByName("Position-X") : 0.3762563132923542,
                squareCapsule.getAttributeByName("Position-Y") : 0.6237436867076458,
                squareCapsule.getAttributeByName("Size") : 0.175,
                squareCapsule.getAttributeByName("Rotation") : 0.125,
                squareCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                squareCapsule.getAttributeByName("Intensity") : 0.6,
                squareCapsule.getAttributeByName("Strength") : 0.07 }, 1.0 )
                
    tri1Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], 
            { triangleCapsule.getAttributeByName("Position-X") : 0.6237436867076458,
                triangleCapsule.getAttributeByName("Position-Y") : 0.3762563132923542,
                triangleCapsule.getAttributeByName("Size") : 0.105,
                triangleCapsule.getAttributeByName("Rotation") : 0.125,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.28 }, 1.0 )
    
    tri2Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], 
            { triangleCapsule.getAttributeByName("Position-X") : 0.2525126265847084,
                triangleCapsule.getAttributeByName("Position-Y") : 0.5,
                triangleCapsule.getAttributeByName("Size") : 0.105,
                triangleCapsule.getAttributeByName("Rotation") : 0.20833333333333331,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.28 }, 1.0 )
    
    tri3Obs = Observation(triangleCapsule, triangleCapsule._routes[0], [], 
            { triangleCapsule.getAttributeByName("Position-X") : 0.5,
                triangleCapsule.getAttributeByName("Position-Y") : 0.7474873734152916,
                triangleCapsule.getAttributeByName("Size") : 0.105,
                triangleCapsule.getAttributeByName("Rotation") : 0.04166667000000002,
                triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                triangleCapsule.getAttributeByName("Intensity") : 0.1,
                triangleCapsule.getAttributeByName("Strength") : 0.28 }, 1.0 )

    rocketCapsule = capsNet.addSemanticCapsule("Rocket", [sqObs, tri2Obs, tri3Obs], 0) 
    shuttleCapsule = capsNet.addSemanticCapsule("Shuttle", [circObs, tri1Obs], 0)   


    # UI Button Function for adding new Capsules
    def addNewSemanticCapsule(name : str, observationList : list):
        print("Training new Capsule '" + name + "' from:")
        for obs in observationList:
            obs.printOutputs(True)

        newCaps = capsNet.addSemanticCapsule(name, observationList)
        


    for i in range(3):
        # Load the test image
        imageReal, width, height = Utility.loadImage("Examples/Ascene" + str(i) + ".png") 

        # Feed-Forward Pass through the Network
        allObs = capsNet.showInput(imageReal, width, height, 1)

        # Draw from the detected data
        imageObserved, semantics, texts = capsNet.generateImage(width, height, allObs, False)

        # Print all Observations
        for capsule in allObs.keys():
            print(str(len(allObs[capsule])) + "x " + capsule.getName())
            for index, obs in enumerate(allObs[capsule]):
                print("Observation " + str(index))
                obs.printOutputs(False)
                    
        testUI.draw(imageReal, imageObserved, width, height, width, height, semantics, texts, addNewSemanticCapsule, False)