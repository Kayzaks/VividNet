

from IntuitivePhysics import IntuitivePhysics
from CapsuleNetwork import CapsuleNetwork
from PrimitivesPhysics import PrimitivesPhysics
from PrimitivesRenderer import PrimitivesRenderer

class VividNet:

    def __init__(self):
        self._capsuleNetwork   : CapsuleNetwork     = CapsuleNetwork()
        self._intuitivePhysics : IntuitivePhysics   = IntuitivePhysics(self._capsuleNetwork)


    def setRenderer(self, primitivesRenderer : PrimitivesRenderer):
        self._capsuleNetwork.setRenderer(primitivesRenderer)
        

    def setPhysics(self, primitivesPhysics : PrimitivesPhysics):
        self._intuitivePhysics.setPhysics(primitivesPhysics)