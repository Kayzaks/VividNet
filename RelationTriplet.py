from Observation import Observation
from HyperParameters import HyperParameters
from AttributePool import AttributePool
from Capsule import Capsule
from CapsuleNetwork import CapsuleNetwork

class RelationTriplet:

    @staticmethod
    def tripletLength():
        return (2 * (HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2) + (2 + 2 * HyperParameters.Dimensions))


    @staticmethod
    def mapAttributes(attributePool : AttributePool, capsule : Capsule, values : list):
        attributes = capsule.getAttributes()
        outputs : dict = {}
        for attr in attributes:
            idx = attributePool.getAttributeOrder(attr)
            if idx > -1:
                outputs[attr] = values[idx]
            else:
                outputs[attr] = 0.0
        return outputs


    @staticmethod
    def generate(senderObservation : Observation, receiverObservation : Observation, attributePool : AttributePool, capsNet : CapsuleNetwork):
        senderOutputs = senderObservation.getOutputsList()
        receiverOutputs = receiverObservation.getOutputsList()
        senderVelocities = senderObservation.getVelocities()
        receiverVelocities = receiverObservation.getVelocities()
        
        # Triplet Format:
        # Sender   -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic
        # Receiver -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic 
        # Relation -- Distance | Degrees-Of-Freedom | Sender Normal | Receiver Normal
        
        totalObjectEntries = (HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2)
        triplet = [0.0] * (2 * totalObjectEntries + (2 + 2 * HyperParameters.Dimensions))

        # Symbols
        if senderObservation.getCapsule().getOrderID() < HyperParameters.MaximumSymbolCount:
            triplet[senderObservation.getCapsule().getOrderID()] = 1.0
        if receiverObservation.getCapsule().getOrderID() < HyperParameters.MaximumSymbolCount:
            triplet[totalObjectEntries + receiverObservation.getCapsule().getOrderID()] = 1.0

        # Attributes / Velocities
        for outputAttribute, outputValue in senderOutputs.items():
            pos = attributePool.getAttributeOrder(outputAttribute)
            if pos > -1:
                triplet[HyperParameters.MaximumSymbolCount + pos] = outputValue
                triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + pos] = senderVelocities[outputAttribute]
        for outputAttribute, outputValue in receiverOutputs.items():
            pos = attributePool.getAttributeOrder(outputAttribute)
            if pos > -1:
                triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + pos] = outputValue
                triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + pos] = receiverVelocities[outputAttribute]


        # Static / Dynamic
        # TODO:
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0

        # Rigid / Elastic
        # TODO:
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 0.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 0.0

        # Distance
        # TODO:
        triplet[2 * totalObjectEntries] = 0.0

        # Degrees-Of-Freedom
        # TODO:
        triplet[2 * totalObjectEntries + 1] = 2.0 / float(HyperParameters.DegreesOfFreedom)

        # Sender Normal
        # TODO:
        triplet[2 * totalObjectEntries + 2] = 1.0
        triplet[2 * totalObjectEntries + 3] = 0.0

        # Receiver Normal
        # TODO:
        triplet[2 * totalObjectEntries + 5] = -1.0
        triplet[2 * totalObjectEntries + 6] = 0.0

        return triplet