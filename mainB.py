
# http://brm.io/matter-js/demo/#mixed
# http://www.pymunk.org/en/master/

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


if __name__ == '__main__':
    testUI = GraphicsUserInterface()

    vividNet = VividNet()

    primCaps = vividNet.setRenderer(TestRenderer, TestPrimitives)
    vividNet.setSyntheticPhysics(TestPhysics) 

    circleCapsule = primCaps[TestPrimitives.Circle]


    attr = { circleCapsule.getAttributeByName("Position-X") :    0.5,
                circleCapsule.getAttributeByName("Position-Y") : 0.4,
                circleCapsule.getAttributeByName("Size") : 0.2,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.8,
                circleCapsule.getAttributeByName("Strength") : 0.1  }
    
    circObs1 = Observation(circleCapsule, circleCapsule._routes[0], [], attr, 1.0 )

    attr = { circleCapsule.getAttributeByName("Position-X") :    0.5,
                circleCapsule.getAttributeByName("Position-Y") : 0.6,
                circleCapsule.getAttributeByName("Size") : 0.2,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.8,
                circleCapsule.getAttributeByName("Strength") : 0.1  }
    
    circObs2 = Observation(circleCapsule, circleCapsule._routes[0], [], attr, 1.0 )
    
    figure8Capsule = vividNet._capsuleNetwork.addSemanticCapsule("Figure8", [circObs1, circObs2], 0) #, 1)   


    shape = (84, 84)            # Total Size of the Image
    
    numFrames = 40 #0
    framesSim = []
    framesReal = []

    #vividNet._intuitivePhysics.trainPhiR(True)
    #vividNet._intuitivePhysics.trainPhiO(True)

    physics = TestPhysics(vividNet._capsuleNetwork.getAttributePool())





    APos = np.array([0.2, 0.5])
    AVel = np.array([0.6, -0.4])
    AMass = 0.3

    BPos = np.array([0.6, 0.5])
    BVel = np.array([0.0, 0.0])
    BMass = 0.2

    CPos = np.array([0.3, 0.85])
    CVel = np.array([0.0, -0.2])
    CMass = 0.2


    obsDicts = []        
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  APos[0],
                circleCapsule.getAttributeByName("Position-Y") :        APos[1],
                circleCapsule.getAttributeByName("Size") : AMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.3,
                circleCapsule.getAttributeByName("Strength") : 0.1 })                    
    obsDicts.append({ figure8Capsule.getAttributeByName("Position-X") :  BPos[0],
                figure8Capsule.getAttributeByName("Position-Y") :        BPos[1],
                figure8Capsule.getAttributeByName("Size") : BMass ,
                figure8Capsule.getAttributeByName("Rotation") : 0.0,
                figure8Capsule.getAttributeByName("Aspect-Ratio") : 1.0,
                figure8Capsule.getAttributeByName("Intensity") : 0.7,
                figure8Capsule.getAttributeByName("Strength") : 0.1 })                
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  CPos[0],
                circleCapsule.getAttributeByName("Position-Y") :        CPos[1],
                circleCapsule.getAttributeByName("Size") : CMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.7,
                circleCapsule.getAttributeByName("Strength") : 0.1 })
    
    obsA1 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[0], 1.0 )
    obsB1 = Observation(figure8Capsule, figure8Capsule._routes[0], [], obsDicts[1], 1.0 )
    obsC1 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[2], 1.0 )
    


    obsDicts = []        
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  APos[0] + AVel[0] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Position-Y") :        APos[1] + AVel[1] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Size") : AMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.3,
                circleCapsule.getAttributeByName("Strength") : 0.1 })                    
    obsDicts.append({ figure8Capsule.getAttributeByName("Position-X") :  BPos[0] + BVel[0] * HyperParameters.TimeStep,
                figure8Capsule.getAttributeByName("Position-Y") :        BPos[1] + BVel[1] * HyperParameters.TimeStep,
                figure8Capsule.getAttributeByName("Size") : BMass ,
                figure8Capsule.getAttributeByName("Rotation") : 0.0,
                figure8Capsule.getAttributeByName("Aspect-Ratio") : 1.0,
                figure8Capsule.getAttributeByName("Intensity") : 0.7,
                figure8Capsule.getAttributeByName("Strength") : 0.1 })                 
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  CPos[0] + CVel[0] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Position-Y") :        CPos[1] + CVel[1] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Size") : CMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.7,
                circleCapsule.getAttributeByName("Strength") : 0.1 })
    
    obsA2 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[0], 1.0 )
    obsB2 = Observation(figure8Capsule, figure8Capsule._routes[0], [], obsDicts[1], 1.0 )
    obsC2 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[2], 1.0 )
    
    obsA2.linkPreviousObservation(obsA1)
    obsB2.linkPreviousObservation(obsB1)
    obsC2.linkPreviousObservation(obsC1)

    simObs = {circleCapsule : [], figure8Capsule : []}

    simObs[circleCapsule].append(obsA2)
    simObs[figure8Capsule].append(obsB2)
    #simObs[circleCapsule].append(obsC2)

    framesSim.append(simObs)
    framesReal.append(simObs)


    for i in range(numFrames):
        simObs = vividNet._intuitivePhysics.predict(simObs, [0.5, 0.5, 0.5])
        framesSim.append(simObs)
        
    
    drawFrames = physics.render(vividNet._capsuleNetwork, framesSim, shape[0], shape[1])

    testUI.drawMovie(drawFrames, shape[0], shape[1], HyperParameters.TimeStep, True)


