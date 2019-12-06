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

    vividNet = VividNet("vividnetB")

    retrainPrimitives = 2       # After initial Training, add 2 more loops, set this to 0 
                                # after completion.
    retrainPhysics = 1          # After initial Training, add 1 more loop, set this to 0 
                                # after completion.
    primCaps = vividNet.setRenderer(TestRenderer, TestPrimitives, retrainPrimitives)
    vividNet.setSyntheticPhysics(TestPhysics, retrainPhysics)

    simObs, ignoreR = vividNet.showFrame("Examples/Bframe1.0.png")
    simObs, ignoreR = vividNet.showFrame("Examples/Bframe1.1.png") 

    # Print all Observations and their continuity
    for capsule in simObs.keys():
        print(str(len(simObs[capsule])) + "x " + capsule.getName())
        for index, obs in enumerate(simObs[capsule]):
            print("Observation " + str(index))
            obs.printContinuity(False)
                    
    
    drawFrames = vividNet.renderPrediction(40)

    testUI.drawMovie(drawFrames, vividNet.getWidth(), vividNet.getHeight(), HyperParameters.TimeStep, False)


