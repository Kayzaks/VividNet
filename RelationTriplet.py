from Observation import Observation
from HyperParameters import HyperParameters
from AttributePool import AttributePool
from Capsule import Capsule
from CapsuleNetwork import CapsuleNetwork

class RelationTriplet:

    @staticmethod
    def tripletLength():
        return (2 * (HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2 * HyperParameters.DegreesOfFreedom) \
             + (1 + HyperParameters.DegreesOfFreedom + 2 * HyperParameters.Dimensions))


    @staticmethod
    def mapAttributes(attributePool : AttributePool, capsule : Capsule, values : list, offset : int = 0):
        attributes = capsule.getAttributes()
        outputs : dict = {}
        for attr in attributes:
            idx = attributePool.getAttributeOrder(attr)
            if idx > -1:
                outputs[attr] = values[idx + offset]
            else:
                outputs[attr] = 0.0
        return outputs


    @staticmethod
    def generate(senderObservation : Observation, receiverObservation : Observation, attributePool : AttributePool, capsNet : CapsuleNetwork):
        senderOutputs = senderObservation.getOutputs()
        receiverOutputs = receiverObservation.getOutputs()
        senderVelocities = senderObservation.getVelocities(HyperParameters.TimeStep)
        receiverVelocities = receiverObservation.getVelocities(HyperParameters.TimeStep)
        
        # Triplet Format:
        # Sender   -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic
        # Receiver -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic 
        # Relation -- Distance | Degrees-Of-Freedom | Sender Normal | Receiver Normal
        
        totalObjectEntries = (HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2 * HyperParameters.DegreesOfFreedom)
        triplet = [0.0] * RelationTriplet.tripletLength()

        # Symbols
        if senderObservation.getCapsule().getOrderID() < HyperParameters.MaximumSymbolCount:
            triplet[senderObservation.getCapsule().getOrderID()] = 1.0
        if receiverObservation.getCapsule().getOrderID() < HyperParameters.MaximumSymbolCount:
            triplet[totalObjectEntries + receiverObservation.getCapsule().getOrderID()] = 1.0

        # Attributes / Velocities
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + i] = 0.5
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + i] = 0.5
        for outputAttribute, outputValue in senderOutputs.items():
            pos = attributePool.getAttributeOrder(outputAttribute)
            if pos > -1:
                triplet[HyperParameters.MaximumSymbolCount + pos] = outputValue
                triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + pos] = (senderVelocities[outputAttribute] + 1.0) * 0.5
        for outputAttribute, outputValue in receiverOutputs.items():
            pos = attributePool.getAttributeOrder(outputAttribute)
            if pos > -1:
                triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + pos] = outputValue
                triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + pos] = (receiverVelocities[outputAttribute] + 1.0) * 0.5


        DQA, DSA = senderObservation.getCapsule().getPhysicalProperties()
        DQB, DSB = receiverObservation.getCapsule().getPhysicalProperties()

        # Static / Dynamic
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = DQA[0]
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = DQA[1]
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2] = DQA[2]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = DQB[0]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = DQB[1]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2] = DQB[2]

        # Rigid / Elastic
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 3] = DSA[0]
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 4] = DSA[1]
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 5] = DSA[2]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 3] = DSB[0]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 4] = DSB[1]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 5] = DSB[2]

        # Distance
        dist, norm1, norm2 = capsNet.distance(senderObservation, receiverObservation)
        triplet[2 * totalObjectEntries] = dist

        # Degrees-Of-Freedom
        # TODO:
        triplet[2 * totalObjectEntries + 1] = 1.0
        triplet[2 * totalObjectEntries + 2] = 1.0
        triplet[2 * totalObjectEntries + 3] = 1.0

        # Normal:
        for i in range(HyperParameters.Dimensions):
            triplet[2 * totalObjectEntries + 1 + HyperParameters.DegreesOfFreedom + i] = (norm1[i] + 1.0) * 0.5
            triplet[2 * totalObjectEntries + 1 + HyperParameters.DegreesOfFreedom + HyperParameters.Dimensions + i] = (norm2[i] + 1.0) * 0.5

        return triplet, dist