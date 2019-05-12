from NeuralNetPhiR import NeuralNetPhiR 
from NeuralNetPhiO import NeuralNetPhiO 
from PrimitivesPhysics import PrimitivesPhysics
from CapsuleNetwork import CapsuleNetwork
from PhysicsMemory import PhysicsMemory
from PhysicsMemory import PhysicsMemoryMode
from RelationTriplet import RelationTriplet
from HyperParameters import HyperParameters
from Observation import Observation
from AttributePool import AttributePool

import numpy as np

class IntuitivePhysics:

    def __init__(self, capsNet : CapsuleNetwork):
        self._neuralNetPhiR    : NeuralNetPhiR     = None
        self._neuralNetPhiO    : NeuralNetPhiO     = None
        self._capsuleNetwork   : CapsuleNetwork    = capsNet
        self._physicsMemory    : PhysicsMemory     = PhysicsMemory()
        
        # Object Attributes + Symbol + Effects + External Effects
        phiOInput = HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount  + HyperParameters.DegreesOfFreedom * 2   

        self._neuralNetPhiR = NeuralNetPhiR(None, None, capsNet.getName() + "-IP-PhiR", False, True, (RelationTriplet.tripletLength(), HyperParameters.DegreesOfFreedom))
        self._neuralNetPhiO = NeuralNetPhiO(None, None, capsNet.getName() + "-IP-PhiO", False, True, (phiOInput, HyperParameters.MaximumAttributeCount))
            

    def aggregate(self, effects : list, external : list, observation : Observation):
        # effects       # List of Effect Vectors
        # external      # External Effect Vector
        # observation   # Original Observation

        aggregated = [0.0] * (HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + HyperParameters.DegreesOfFreedom * 2)
        
        for attr, value in observation.getOutputsList().items():
            pos = self._capsuleNetwork.getAttributePool().getAttributeOrder(attr)
            if pos > -1:
                aggregated[pos] = value

        if observation.getCapsule().getOrderID() < HyperParameters.MaximumSymbolCount:
            aggregated[HyperParameters.MaximumAttributeCount + observation.getCapsule().getOrderID()] = 1.0

        for attr, value in observation.getVelocities().items():
            pos = self._capsuleNetwork.getAttributePool().getAttributeOrder(attr)
            if pos > -1:
                aggregated[HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + pos] = value

        for i in range(HyperParameters.DegreesOfFreedom):
            summed = 0.0
            for effect in effects:
                summed = summed + effect[i]
            aggregated[HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + i] = summed      
            
        for i in range(HyperParameters.DegreesOfFreedom):
            aggregated[HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount  + HyperParameters.DegreesOfFreedom + i] = external[i]

        return aggregated     # Object Attributes + Symbol + Velocities + Effects + External Effects


    def fillMemorySynthetically(self, primitivesPhysics : PrimitivesPhysics):
        self._physicsMemory.setSyntheticPhysics(primitivesPhysics)


    def train(self, showDebugOutput : bool = True):
        self._physicsMemory.setMode(PhysicsMemoryMode.PhiR)
        self._neuralNetPhiR.trainFromData(self._physicsMemory, showDebugOutput)
        self._physicsMemory.setMode(PhysicsMemoryMode.PhiO)
        self._neuralNetPhiO.trainFromData(self._physicsMemory, showDebugOutput)


    def predict(self, observations : dict, external : list):
        
        newObservations = {}

        for caps, obsList in observations.items():
            newObservations[caps] = []
            for receiverObs in obsList:
                effects = []
                for senderObs in [x for senderObsList in observations.values() for x in senderObsList]:
                    if senderObs != receiverObs:
                        triplet = RelationTriplet.generate(senderObs, receiverObs, self._capsuleNetwork.getAttributePool(), self._capsuleNetwork)
                        effects.append(self._neuralNetPhiR.forwardPass(triplet))
                
                aggregated = self.aggregate(effects, external, receiverObs)
                result = self._neuralNetPhiO.forwardPass(aggregated)

                attributes = RelationTriplet.mapAttributes(self._capsuleNetwork.getAttributePool(), caps, result)
                newObs = Observation(caps, receiverObs.getTakenRoute(), receiverObs.getInputObservations(), attributes, receiverObs.getProbability())

                newObservations[caps].append(newObs)

        return newObservations

