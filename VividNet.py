

from IntuitivePhysics import IntuitivePhysics
from CapsuleNetwork import CapsuleNetwork
from PrimitivesPhysics import PrimitivesPhysics
from PrimitivesRenderer import PrimitivesRenderer

class VividNet:

    def __init__(self):
        self._capsuleNetwork   : CapsuleNetwork     = CapsuleNetwork()
        self._intuitivePhysics : IntuitivePhysics   = IntuitivePhysics(self._capsuleNetwork)


    def setRenderer(self, rendererClass, primitivesEnum, extraTraining : int = 0):
        self._capsuleNetwork.setRenderer(rendererClass)

        capsuleList = {}
        dimensions = self._capsuleNetwork.getRenderer().getDimensions()
        for primType in primitivesEnum:
            if int(primType) > -1:
                capsuleList[primType] = self._capsuleNetwork.addPrimitiveCapsule(primType, dimensions[primType], extraTraining) 

        return capsuleList

 
    def setSyntheticPhysics(self, primitivesPhysics : PrimitivesPhysics):
        self._intuitivePhysics.fillMemorySynthetically(primitivesPhysics)