
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
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#set_session(tf.Session(config=config))


if __name__ == '__main__':
    testUI = GraphicsUserInterface()

    vividNet = VividNet()

    primCaps = vividNet.setRenderer(TestRenderer, TestPrimitives)
    vividNet.setSyntheticPhysics(TestPhysics) 

    circleCapsule = primCaps[TestPrimitives.Circle]

    shape = (84, 84)            # Total Size of the Image
    
    numFrames = 30 #0
    framesSim = []
    framesReal = []

    #vividNet._intuitivePhysics.trainPhiR(True)
    #vividNet._intuitivePhysics.trainPhiO(True)

    physics = TestPhysics(vividNet._capsuleNetwork.getAttributePool())



    APos = np.array([0.2, 0.2])
    AVel = np.array([0.4, 0.4])
    AMass = 0.2

    BPos = np.array([0.5, 0.5])
    BVel = np.array([0.0, 0.0])
    BMass = 0.2

    CPos = np.array([0.8, 0.8])
    CVel = np.array([0.0, 0.0])
    CMass = 0.2

    simAPos = np.array([APos[0], APos[1]])
    simAVel = np.array([AVel[0], AVel[1]])
    simBPos = np.array([BPos[0], BPos[1]])
    simBVel = np.array([BVel[0], BVel[1]])
    simCPos = np.array([CPos[0], CPos[1]])
    simCVel = np.array([CVel[0], CVel[1]])

    
    obsDicts = []        
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  simAPos[0],
                circleCapsule.getAttributeByName("Position-Y") :        simAPos[1],
                circleCapsule.getAttributeByName("Size") : AMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.3,
                circleCapsule.getAttributeByName("Strength") : 0.1 })                    
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  simBPos[0],
                circleCapsule.getAttributeByName("Position-Y") :        simBPos[1],
                circleCapsule.getAttributeByName("Size") : BMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.7,
                circleCapsule.getAttributeByName("Strength") : 0.1 })                
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  simCPos[0],
                circleCapsule.getAttributeByName("Position-Y") :        simCPos[1],
                circleCapsule.getAttributeByName("Size") : CMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.7,
                circleCapsule.getAttributeByName("Strength") : 0.1 })
    
    obsA1 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[0], 1.0 )
    obsB1 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[1], 1.0 )
    obsC1 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[2], 1.0 )
    
    simObs = {circleCapsule : []}
    obsDicts = []        
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  simAPos[0] + AVel[0] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Position-Y") :        simAPos[1] + AVel[1] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Size") : AMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.3,
                circleCapsule.getAttributeByName("Strength") : 0.1 })                    
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  simBPos[0] + BVel[0] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Position-Y") :        simBPos[1] + BVel[1] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Size") : BMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.7,
                circleCapsule.getAttributeByName("Strength") : 0.1 })                 
    obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  simCPos[0] + CVel[0] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Position-Y") :        simCPos[1] + CVel[1] * HyperParameters.TimeStep,
                circleCapsule.getAttributeByName("Size") : CMass ,
                circleCapsule.getAttributeByName("Rotation") : 0.0,
                circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                circleCapsule.getAttributeByName("Intensity") : 0.7,
                circleCapsule.getAttributeByName("Strength") : 0.1 })
    
    obsA2 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[0], 1.0 )
    obsB2 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[1], 1.0 )
    obsC2 = Observation(circleCapsule, circleCapsule._routes[0], [], obsDicts[2], 1.0 )
    
    obsA2.linkPreviousObservation(obsA1)
    obsB2.linkPreviousObservation(obsB1)
    obsC2.linkPreviousObservation(obsC1)

    simObs[circleCapsule].append(obsA2)
    simObs[circleCapsule].append(obsB2)
    simObs[circleCapsule].append(obsC2)

    framesSim.append(simObs)
    framesReal.append(simObs)


    for i in range(numFrames):
        
        timeStep = HyperParameters.TimeStep * float(i) 

        ignoredB, realEffectB = physics.generateTestRelation(APos[0], APos[1], AVel[0], AVel[1], AMass, BPos[0], BPos[1], BVel[0], BVel[1], BMass)
        ignoredA, realEffectA = physics.generateTestRelation(BPos[0], BPos[1], BVel[0], BVel[1], BMass, APos[0], APos[1], AVel[0], AVel[1], AMass)

        AAcc = (np.array([realEffectA[0], realEffectA[1]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale
        AOldPos = APos
        APos = APos + AVel * HyperParameters.TimeStep + 0.5 * AAcc * HyperParameters.TimeStep * HyperParameters.TimeStep
        AVel = (APos - AOldPos) / HyperParameters.TimeStep + 0.5 * AAcc * HyperParameters.TimeStep
        
        BAcc = (np.array([realEffectB[0], realEffectB[1]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale
        BOldPos = BPos
        BPos = BPos + BVel * HyperParameters.TimeStep + 0.5 * BAcc * HyperParameters.TimeStep * HyperParameters.TimeStep
        BVel = (BPos - BOldPos) / HyperParameters.TimeStep + 0.5 * BAcc * HyperParameters.TimeStep 


        realObs = {circleCapsule : []}        
        obsDicts = []        
        obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  APos[0],
                    circleCapsule.getAttributeByName("Position-Y") :        APos[1],
                    circleCapsule.getAttributeByName("Size") : AMass ,
                    circleCapsule.getAttributeByName("Rotation") : 0.0,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.3,
                    circleCapsule.getAttributeByName("Strength") : 0.1 })                    
        obsDicts.append({ circleCapsule.getAttributeByName("Position-X") :  BPos[0],
                    circleCapsule.getAttributeByName("Position-Y") :        BPos[1],
                    circleCapsule.getAttributeByName("Size") : BMass ,
                    circleCapsule.getAttributeByName("Rotation") : 0.0,
                    circleCapsule.getAttributeByName("Aspect-Ratio") : 1.0,
                    circleCapsule.getAttributeByName("Intensity") : 0.7,
                    circleCapsule.getAttributeByName("Strength") : 0.1 })
        
        for obs in obsDicts:
            realObs[circleCapsule].append(Observation(circleCapsule, circleCapsule._routes[0], [], obs, 1.0 ))

        framesReal.append(realObs)

        simObs = vividNet._intuitivePhysics.predict(simObs, [0.5, 0.5, 0.5])
        framesSim.append(simObs)
        
    
    #drawFrames = physics.render(vividNet._capsuleNetwork, framesReal, shape[0], shape[1])

    #testUI.drawMovie(drawFrames, shape[0], shape[1], HyperParameters.TimeStep)

    
    drawFrames = physics.render(vividNet._capsuleNetwork, framesSim, shape[0], shape[1])

    testUI.drawMovie(drawFrames, shape[0], shape[1], HyperParameters.TimeStep)


