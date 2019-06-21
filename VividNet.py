

from IntuitivePhysics import IntuitivePhysics
from CapsuleNetwork import CapsuleNetwork
from PrimitivesPhysics import PrimitivesPhysics
from PrimitivesRenderer import PrimitivesRenderer
from Utility import Utility
from HyperParameters import HyperParameters

from pathlib import Path

import json
import math

class VividNet:

    def __init__(self, vnName : str):
        self._name             : str                = vnName
        self._capsuleNetwork   : CapsuleNetwork     = CapsuleNetwork(vnName + "-CN")
        self._intuitivePhysics : IntuitivePhysics   = IntuitivePhysics(self._capsuleNetwork)
        self._pastFrames       : list               = []                                        # List of {Capsule - List of Observations}
        self._inputWidth       : int                = 1                                         # Last Frame Width
        self._inputHeight      : int                = 1                                         # Last Frame Height

    def getJSON(self):
        return {"name"           : self._name,
                "pastFrames"     : self._pastFrames,
                "inputWidth"     : self._inputWidth,
                "inputHeight"    : self._inputHeight,
                "capsuleNetwork" : self._capsuleNetwork.getJSON()}

    def putJSON(self, data):
        self._name = data["name"]
        self._pastFrames = data["pastFrames"]
        self._inputWidth = data["inputWidth"]
        self._inputHeight = data["inputHeight"]
        return self._capsuleNetwork.putJSON(data["capsuleNetwork"])   # List of Semantic Capsules


    def toJSON(self):
        return json.dumps(self, default=lambda o: o.getJSON(), sort_keys=True, indent=4)

    def loadSemantic(self):
        fpath = Path(self._name + ".json")
        if fpath.is_file():
            with open(self._name + ".json", encoding="utf-8") as inputfile:  
                data = json.load(inputfile)
                return self.putJSON(data)

        return {}

    def saveSemantic(self):
        with open(self._name + ".json", "w", encoding="utf-8") as outfile:
            outfile.write(self.toJSON())


    def getWidth(self):
        return self._inputWidth

    def getHeight(self):
        return self._inputHeight


    def setRenderer(self, rendererClass, primitivesEnum, extraTraining : int = 0):
        self._capsuleNetwork.setRenderer(rendererClass)

        capsuleList = {}
        dimensions = self._capsuleNetwork.getRenderer().getDimensions()
        for primType in primitivesEnum:
            if int(primType) > -1:
                capsuleList[primType] = self._capsuleNetwork.addPrimitiveCapsule(primType, dimensions[primType], extraTraining) 

        return capsuleList

 
    def setSyntheticPhysics(self, primitivesPhysics : PrimitivesPhysics, extraTraining : int = 0):
        self._intuitivePhysics.fillMemorySynthetically(primitivesPhysics, extraTraining)


    def showFrame(self, filename : str):
        imageReal, self._inputWidth, self._inputHeight = Utility.loadImage(filename) 
        simObs = self._capsuleNetwork.showInput(imageReal, self._inputWidth, self._inputHeight, 1)

        self.applyContinuity(simObs)
        return simObs

    
    def applyContinuity(self, newObservations : dict):
        # newObservations   # {Capsule - List of Observations}

        maxPastFrames = 1
        
        # TODO: The search in the "far past", i.e. for objects that 
        #       Lost occlusion long ago is not correct. 
        #       Velocities get calculated wrong and need to be aggregated
        #       (also in Observation class)

        for i in range(maxPastFrames):
            if i >= len(self._pastFrames):
                break
                
            # Start with Frames in Reverse Order
            lastFrame = self._pastFrames[len(self._pastFrames) - 1 - i]
            for capsule, obsList in newObservations.items():
                if capsule in lastFrame:
                    for obs in obsList:
                        if obs.hasPreviousObservation() is False:
                            attributes = obs.getOutputs()
                            
                            for pastObs in lastFrame[capsule]:
                                pastAttributes = pastObs.getOutputs()
                                pastVelocity = pastObs.getVelocities(HyperParameters.TimeStep)

                                totalDiff = 0.0
                                for attr, value in attributes.items():
                                    attrDiff = pastAttributes[attr] + pastVelocity[attr] * HyperParameters.TimeStep * float(i) - value
                                    totalDiff = totalDiff + attrDiff * attrDiff

                                # TODO: Weight Vector
                                if math.sqrt(totalDiff) < HyperParameters.ContinuityCutoff:
                                    # We have continuity!
                                    obs.linkPreviousObservation(pastObs)
                                    break
                       
        self._pastFrames.append(newObservations)


    def renderPrediction(self, numFrames : int):
        if len(self._pastFrames) < 1:
            return 0

        framesPixels = []
        observationFrame = self._pastFrames[-1]

        for i in range(numFrames):
            imageReal, semantics, texts = self._capsuleNetwork.generateImage(self._inputWidth, self._inputHeight, observationFrame, False)

            framesPixels.append(imageReal)

            observationFrame = self._intuitivePhysics.predict(observationFrame, [0.5, 0.5, 0.5])

        return framesPixels

