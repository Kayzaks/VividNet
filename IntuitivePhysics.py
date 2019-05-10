from NeuralNetPhiR import NeuralNetPhiR 
from NeuralNetPhiO import NeuralNetPhiO 
from PrimitivesPhysics import PrimitivesPhysics
from CapsuleNetwork import CapsuleNetwork

class IntuitivePhysics:

    def __init__(self, capsNet : CapsuleNetwork):
        self._neuralNetPhiR    : NeuralNetPhiR     = None
        self._neuralNetPhiO    : NeuralNetPhiO     = None
        self._capsuleNetwork   : CapsuleNetwork    = capsNet
        self._synthPhysics     : PrimitivesPhysics = None


    def setPhysics(self, primitivesPhysics : PrimitivesPhysics):
        self._synthPhysics = primitivesPhysics


    def trainSynthetically(self):

        return 0