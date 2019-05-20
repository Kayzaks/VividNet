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
        
        self._neuralNetPhiR = NeuralNetPhiR(None, None, capsNet.getName() + "-IP-PhiR", False, True, (RelationTriplet.tripletLength(), HyperParameters.DegreesOfFreedom))
        self._neuralNetPhiO = NeuralNetPhiO(None, None, capsNet.getName() + "-IP-PhiO", False, True, (self.getAggregateDimension(), HyperParameters.MaximumAttributeCount * 2))
            

    def getAggregateDimension(self):
        return HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + HyperParameters.DegreesOfFreedom * 4
    

    def aggregate(self, effects : list, external : list, observation : Observation):
        # effects       # List of Effect Vectors
        # external      # External Effect Vector
        # observation   # Original Observation

        aggregated = [0.0] * self.getAggregateDimension()
        
        # Attributes
        for attr, value in observation.getOutputs().items():
            pos = self._capsuleNetwork.getAttributePool().getAttributeOrder(attr)
            if pos > -1:
                aggregated[pos] = value

        # Symbols
        if observation.getCapsule().getOrderID() < HyperParameters.MaximumSymbolCount:
            aggregated[HyperParameters.MaximumAttributeCount + observation.getCapsule().getOrderID()] = 1.0

        # Velocities
        for i in range(HyperParameters.MaximumAttributeCount):
            aggregated[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + i] = 0.5

        for attr, value in observation.getVelocities(HyperParameters.TimeStep).items():
            pos = self._capsuleNetwork.getAttributePool().getAttributeOrder(attr)
            if pos > -1:
                aggregated[HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + pos] = (value + 1.0) * 0.5

        # Static/Dynamic + Rigid/Elastic
        DQ, DS = observation.getCapsule().getPhysicalProperties()
        aggregated[HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount] = DQ[0]
        aggregated[HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + 1] = DQ[1]
        aggregated[HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + 2] = DQ[2]

        aggregated[HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + 3] = DS[0]
        aggregated[HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + 4] = DS[1]
        aggregated[HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + 5] = DS[2]

        # Effects
        offset = HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.DegreesOfFreedom
        for i in range(HyperParameters.DegreesOfFreedom):
            summed = 0.0
            for effect in effects:
                summed = summed + ((effect[i] * 2.0) - 1.0)
            aggregated[offset + i] = (summed + 1.0) * 0.5   
            
        # External Effects
        for i in range(HyperParameters.DegreesOfFreedom):
            aggregated[offset + HyperParameters.DegreesOfFreedom + i] = external[i]

        return aggregated     # Object Attributes + Symbol + Velocities + Static/Dynamic + Rigid/Elastic + Effects + External Effects


    def fillMemorySynthetically(self, primitivesPhysics : PrimitivesPhysics):
        self._physicsMemory.setSyntheticPhysics(primitivesPhysics, self._capsuleNetwork.getAttributePool())

        if self._neuralNetPhiR.hasTraining() is False:
            self.trainPhiR()
        if self._neuralNetPhiO.hasTraining() is False:
            self.trainPhiO()


    def trainPhiR(self, showDebugOutput : bool = True):
        self._physicsMemory.setMode(PhysicsMemoryMode.PhiR)
        self._neuralNetPhiR.trainFromData(self._physicsMemory, showDebugOutput)


    def trainPhiO(self, showDebugOutput : bool = True):
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
                        triplet, distance = RelationTriplet.generate(senderObs, receiverObs, self._capsuleNetwork.getAttributePool(), self._capsuleNetwork)
                        effects.append(self._neuralNetPhiR.forwardPass(triplet))
                
                aggregated = self.aggregate(effects, external, receiverObs)

                result = self._neuralNetPhiO.forwardPass(aggregated)

                attributes = RelationTriplet.mapAttributes(self._capsuleNetwork.getAttributePool(), caps, result)
                accelerations = RelationTriplet.mapAttributes(self._capsuleNetwork.getAttributePool(), caps, result, HyperParameters.MaximumAttributeCount)
                
                newObs = Observation(caps, receiverObs.getTakenRoute(), receiverObs.getInputObservations(), attributes, receiverObs.getProbability())
                newObs.linkPreviousObservation(receiverObs)
                
                for attr, accel in accelerations.items():
                    accelerations[attr] = ((accel * 2.0) - 1.0) * HyperParameters.AccelerationScale

                newObs.setAccelerations(accelerations)

                newObservations[caps].append(newObs)

            

        return newObservations

