
# http://brm.io/matter-js/demo/#mixed
# http://www.pymunk.org/en/master/

from TestPrimitives import TestPrimitives
from TestPrimitives import TestRenderer
from TestPrimitives import TestPhysics
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
    circleCapsule = capsNet.addPrimitiveCapsule(TestPrimitives.Circle, [(width, height)], 0) #, 1) 

    numFrames = 10 #0
    frames = []

    for i in range(numFrames):
        
        timeStep = 0.01 * float(i) 

        allObs = {circleCapsule : []}
        
        obsDicts = []
        
        obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  0.3 + timeStep,
                    circleCapsule.getAttributeByName("Position-Y") :        0.5,
                    circleCapsule.getAttributeByName("Size") : 0.1 ,
                    circleCapsule.getAttributeByName("Rotation") : 0.0,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.266,
                    circleCapsule.getAttributeByName("Strength") : 0.3 })
                    
        obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  0.8 - timeStep,
                    circleCapsule.getAttributeByName("Position-Y") :        0.5,
                    circleCapsule.getAttributeByName("Size") : 0.1 ,
                    circleCapsule.getAttributeByName("Rotation") : 0.0,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.266,
                    circleCapsule.getAttributeByName("Strength") : 0.3 })
        
        for obs in obsDicts:
            allObs[circleCapsule].append(Observation(circleCapsule, circleCapsule._routes[0], [], obs, 1.0 ))

        frames.append(allObs)


    physics = TestPhysics()

    drawFrames = physics.render(capsNet, frames, shape[0], shape[1])

    testUI.drawMovie(drawFrames, shape[0], shape[1], 24)

