
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

    shape = (84, 84)            # Total Size of the Image
    
    numFrames = 30 #0
    framesSim = []
    framesReal = []

    #vividNet._intuitivePhysics.trainPhiR(True)
    #vividNet._intuitivePhysics.trainPhiO(True)

    physics = TestPhysics(vividNet._capsuleNetwork.getAttributePool())



    APos = np.array([0.2, 0.2])
    AVel = np.array([0.6, 0.6])
    AMass = 0.2

    BPos = np.array([0.7, 0.7])
    BVel = np.array([-0.4, 0.0])
    BMass = 0.3

    simAPos = np.array([APos[0], APos[1]])
    simAVel = np.array([AVel[0], AVel[1]])
    simBPos = np.array([BPos[0], BPos[1]])
    simBVel = np.array([BVel[0], BVel[1]])


    for i in range(numFrames):
        
        timeStep = HyperParameters.TimeStep * float(i) 

        ignored, realEffectB = physics.generateTestRelation(APos[0], APos[1], AVel[0], AVel[1], AMass, BPos[0], BPos[1], BVel[0], BVel[1], BMass)
        ignored, realEffectA = physics.generateTestRelation(BPos[0], BPos[1], BVel[0], BVel[1], BMass, APos[0], APos[1], AVel[0], AVel[1], AMass)

        simTripletB, ignoredB = physics.generateTestRelation(simAPos[0], simAPos[1], simAVel[0], simAVel[1], AMass, simBPos[0], simBPos[1], simBVel[0], simBVel[1], BMass)
        simEffectB = vividNet._intuitivePhysics._neuralNetPhiR.forwardPass(simTripletB)
        simTripletA, ignoredA = physics.generateTestRelation(simBPos[0], simBPos[1], simBVel[0], simBVel[1], BMass, simAPos[0], simAPos[1], simAVel[0], simAVel[1], AMass)
        simEffectA = vividNet._intuitivePhysics._neuralNetPhiR.forwardPass(simTripletA)


        allObs = {circleCapsule : []}        
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
            allObs[circleCapsule].append(Observation(circleCapsule, circleCapsule._routes[0], [], obs, 1.0 ))

        framesReal.append(allObs)

        
        allObs = {circleCapsule : []}
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
        
        for obs in obsDicts:
            allObs[circleCapsule].append(Observation(circleCapsule, circleCapsule._routes[0], [], obs, 1.0 ))

        framesSim.append(allObs)

        AAcc = (np.array([realEffectA[0], realEffectA[1]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale
        APos = APos + AVel * HyperParameters.TimeStep + 0.5 * AAcc * HyperParameters.TimeStep * HyperParameters.TimeStep
        AVel = AVel + AAcc * HyperParameters.TimeStep 
        
        BAcc = (np.array([realEffectB[0], realEffectB[1]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale
        BPos = BPos + BVel * HyperParameters.TimeStep + 0.5 * BAcc * HyperParameters.TimeStep * HyperParameters.TimeStep
        BVel = BVel + BAcc * HyperParameters.TimeStep

        aggregate, ignoreA = physics.generateTestInteraction(simTripletA, simEffectA)
        outAttributesA = vividNet._intuitivePhysics._neuralNetPhiO.forwardPass(aggregate)
        aggregate, ignoreB = physics.generateTestInteraction(simTripletB, simEffectB)
        outAttributesB = vividNet._intuitivePhysics._neuralNetPhiO.forwardPass(aggregate)

        simAAcc = (np.array([simEffectA[0], simEffectA[1]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale
        simAVel[0] = ((outAttributesA[physics._xPosOffset] - simAPos[0]) / HyperParameters.TimeStep)
        simAVel[1] = ((outAttributesA[physics._yPosOffset] - simAPos[1]) / HyperParameters.TimeStep)
        simAPos[0] = outAttributesA[physics._xPosOffset]
        simAPos[1] = outAttributesA[physics._yPosOffset]

        simBAcc = (np.array([simEffectB[0], simEffectB[1]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale
        simBVel[0] = ((outAttributesB[physics._xPosOffset] - simBPos[0]) / HyperParameters.TimeStep)
        simBVel[1] = ((outAttributesB[physics._yPosOffset] - simBPos[1]) / HyperParameters.TimeStep)
        simBPos[0] = outAttributesB[physics._xPosOffset]
        simBPos[1] = outAttributesB[physics._yPosOffset]

        print(simEffectA)
        print(simEffectB)



    #drawFrames = physics.render(vividNet._capsuleNetwork, framesReal, shape[0], shape[1])

    #testUI.drawMovie(drawFrames, shape[0], shape[1], HyperParameters.TimeStep)

    
    drawFrames = physics.render(vividNet._capsuleNetwork, framesSim, shape[0], shape[1])

    testUI.drawMovie(drawFrames, shape[0], shape[1], HyperParameters.TimeStep)


