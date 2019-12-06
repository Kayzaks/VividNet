
# ------------------------------------------------------------------
#
# Note1: This is highly un-optimized. It may take several minutes to 
#        process a small image. We split the detection of Figure 5
#        into multiple parts for this reason, however, this has no
#        effect on the algorithm itself.
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
from VividNet import VividNet
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

    retrain = 0                 # After initial Training (2 loops), set this to 0 to avoid
                                # retraining each run

    vividNet = VividNet("vividnetA")
    primCaps = vividNet.setRenderer(TestRenderer, TestPrimitives, retrain)
    semCaps = vividNet.loadSemantic()

    # UI Button Functions for Meta-Learning
    def addNewSemanticCapsule(nameCaps : str, observationList : list):
        print("Training new Capsule '" + nameCaps + "' from:")
        for obs in observationList:
            obs.printOutputs(True)

        resCaps = vividNet._capsuleNetwork.addSemanticCapsule(nameCaps, observationList)
        vividNet.saveSemantic()
        
    def trainExistingSemanticCapsule(nameCaps : str, observationList : list):
        print("Training existing Capsule '" + nameCaps + "' from:")
        for obs in observationList:
            obs.printOutputs(True)

        resCaps = vividNet._capsuleNetwork.addSemanticTraining(nameCaps, observationList)
        vividNet.saveSemantic()
        
    def addNewAttribute(nameCaps : str, nameAttr : str, observationList : list):
        print("Training new Attribute '" + nameAttr + "' for '" + nameCaps + "' from:")
        for obs in observationList:
            obs.printOutputs(True)

        resCaps = vividNet._capsuleNetwork.addAttribute(nameCaps, nameAttr, observationList)
        vividNet.saveSemantic()
        
    def trainExistingAttribute(nameCaps : str, nameAttr : str, observationList : list):
        print("Training existing Attribute '" + nameAttr + "' for '" + nameCaps + "' from:")
        for obs in observationList:
            obs.printOutputs(True)

        resCaps = vividNet._capsuleNetwork.addAttributeTraining(nameCaps, nameAttr, observationList)
        vividNet.saveSemantic()
        

    for i in range(3):
        # Load the test image
        imageReal, width, height = Utility.loadImage("Examples/Ascene" + str(i) + ".png") 

        # Feed-Forward Pass through the Network
        allObs, recommendation = vividNet._capsuleNetwork.showInput(imageReal, width, height, 1)

        # Draw from the detected data
        imageObserved, semantics, texts = vividNet._capsuleNetwork.generateImage(width, height, allObs, False)

        # Print all Observations
        for capsule in allObs.keys():
            print(str(len(allObs[capsule])) + "x " + capsule.getName())
            for index, obs in enumerate(allObs[capsule]):
                print("Observation " + str(index))
                obs.printOutputs(False)
                    
        testUI.draw(imageReal, imageObserved, width, height, width, height, semantics, texts, 
                    addNewSemanticCapsule, trainExistingSemanticCapsule, addNewAttribute, trainExistingAttribute, 
                    False, recommendation)