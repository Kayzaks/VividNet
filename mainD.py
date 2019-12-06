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
from TestPhysics import TestPhysics
from CapsuleNetwork import CapsuleNetwork
from HyperParameters import HyperParameters
from VividNet import VividNet
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

# The Primitive Capsules need to be more exact for this, thus we need (1 + 2) training loops
# and tighter HyperParameters. This is to avoid double detection.
HyperParameters.PrimitiveProbabilityCutOff = 0.80
HyperParameters.SemanticProbabilityCutOff  = 0.85

if __name__ == '__main__':
    testUI = GraphicsUserInterface()

    vividNet = VividNet("vividnetD")

    retrainPrimitives = 0       # After initial Training, add 2 more loops, set this to 0 
                                # after completion.
    retrainPhysics = 1          # After initial Training, add 1 more loop, set this to 0 
                                # after completion.
    primCaps = vividNet.setRenderer(TestRenderer, TestPrimitives, retrainPrimitives)
    vividNet.setSyntheticPhysics(TestPhysics, retrainPhysics)

    # gameObservations, ignoreR = vividNet.showFrame("Examples/Dframe0.0.png")

    gameRunning = True
    currentFrameId = 0

    width = 84
    height = 84

    circleCapsule = vividNet._capsuleNetwork._primitiveCapsules[0]
    triangleCapsule = vividNet._capsuleNetwork._primitiveCapsules[2]

    circObs = []
    triObs = []

    circObs.append(Observation(circleCapsule, circleCapsule._routes[0], [], 
                { circleCapsule.getAttributeByName("Position-X") : 0.1,
                    circleCapsule.getAttributeByName("Position-Y") : 0.2,
                    circleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                    circleCapsule.getAttributeByName("Rotation") : 0.125,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.8,
                    circleCapsule.getAttributeByName("Strength") : 0.07 }, 1.0 ))
                    
    circObs.append(Observation(circleCapsule, circleCapsule._routes[0], [], 
                { circleCapsule.getAttributeByName("Position-X") : 0.3,
                    circleCapsule.getAttributeByName("Position-Y") : 0.8,
                    circleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                    circleCapsule.getAttributeByName("Rotation") : 0.125,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.8,
                    circleCapsule.getAttributeByName("Strength") : 0.07 }, 1.0 ))
                    
    circObs.append(Observation(circleCapsule, circleCapsule._routes[0], [], 
                { circleCapsule.getAttributeByName("Position-X") : 0.9,
                    circleCapsule.getAttributeByName("Position-Y") : 0.2,
                    circleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                    circleCapsule.getAttributeByName("Rotation") : 0.125,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.8,
                    circleCapsule.getAttributeByName("Strength") : 0.07 }, 1.0 ))

                    
    triObs.append(Observation(triangleCapsule, triangleCapsule._routes[0], [], 
                { triangleCapsule.getAttributeByName("Position-X") : 0.5,
                    triangleCapsule.getAttributeByName("Position-Y") : 0.5,
                    triangleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                    triangleCapsule.getAttributeByName("Rotation") : 0.125,
                    triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    triangleCapsule.getAttributeByName("Intensity") : 0.8,
                    triangleCapsule.getAttributeByName("Strength") : 0.15 }, 1.0 ))

    gameObservations = {circleCapsule : circObs, triangleCapsule : triObs}
    vividNet.applyContinuity(gameObservations)

    
    circObs = []
    triObs = []

    circObs.append(Observation(circleCapsule, circleCapsule._routes[0], [], 
                { circleCapsule.getAttributeByName("Position-X") : 0.1,
                    circleCapsule.getAttributeByName("Position-Y") : 0.25,
                    circleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                    circleCapsule.getAttributeByName("Rotation") : 0.125,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.8,
                    circleCapsule.getAttributeByName("Strength") : 0.07 }, 1.0 ))
                    
    circObs.append(Observation(circleCapsule, circleCapsule._routes[0], [], 
                { circleCapsule.getAttributeByName("Position-X") : 0.275,
                    circleCapsule.getAttributeByName("Position-Y") : 0.75,
                    circleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                    circleCapsule.getAttributeByName("Rotation") : 0.125,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.8,
                    circleCapsule.getAttributeByName("Strength") : 0.07 }, 1.0 ))
                    
    circObs.append(Observation(circleCapsule, circleCapsule._routes[0], [], 
                { circleCapsule.getAttributeByName("Position-X") : 0.85,
                    circleCapsule.getAttributeByName("Position-Y") : 0.2,
                    circleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                    circleCapsule.getAttributeByName("Rotation") : 0.125,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.8,
                    circleCapsule.getAttributeByName("Strength") : 0.07 }, 1.0 ))

                    
    triObs.append(Observation(triangleCapsule, triangleCapsule._routes[0], [], 
                { triangleCapsule.getAttributeByName("Position-X") : 0.5,
                    triangleCapsule.getAttributeByName("Position-Y") : 0.5,
                    triangleCapsule.getAttributeByName("Size") : 0.13999999999999999,
                    triangleCapsule.getAttributeByName("Rotation") : 0.125,
                    triangleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    triangleCapsule.getAttributeByName("Intensity") : 0.8,
                    triangleCapsule.getAttributeByName("Strength") : 0.15 }, 1.0 ))

    gameObservations = {circleCapsule : circObs, triangleCapsule : triObs}

    vividNet._inputWidth = width
    vividNet._inputHeight = height
   
    xAttr = triangleCapsule.getAttributeByName("Position-X")
    yAttr = triangleCapsule.getAttributeByName("Position-Y")
    rAttr = triangleCapsule.getAttributeByName("Rotation")

    def dirArrow(direction):
        print(direction)
        xPos = gameObservations[triangleCapsule][0].getOutput(xAttr)
        yPos = gameObservations[triangleCapsule][0].getOutput(yAttr)
        rot = gameObservations[triangleCapsule][0].getOutput(rAttr)

        gameObservations[triangleCapsule][0].setOutput(xAttr, xPos + direction[0] * 0.1)
        gameObservations[triangleCapsule][0].setOutput(yAttr, yPos + direction[1] * 0.1 )
        #gameObservations[triangleCapsule][0].setOutput(rAttr, )
        
    while gameRunning is True:

        # This stores it in episodic memory
        vividNet.applyContinuity(gameObservations)


        currentFrame, gameObservations = vividNet.renderPrediction(1)
        testUI.drawGame(currentFrame[0], width, height, dirArrow, currentFrameId)

        currentFrameId = currentFrameId + 1

