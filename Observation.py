from Attribute import Attribute
from CapsuleRoute import CapsuleRoute

class Observation:

    def __init__(self, route : CapsuleRoute, inputAttributes : dict, outputAttributes : dict, inputProbabilities : dict, outputProbability : float):
        # inputAttributes                                    # Capsule - {Attribute - Value}
        # outputAttributes                                   # Attribute - Value
        # inputProbabilities                                 # Capsule - Probability
        self._inputProbability      : dict          = {}     # Capsule - Probability
        self._inputAttributes       : dict          = {}     # Capsule - {Attribute - Value}
        self._outputAttributes      : dict          = {}     # Attribute - Value
        self._outputProbability     : float         = outputProbability
        self._route                 : CapsuleRoute  = route

        if inputProbabilities is not None:
            for capsule, probability in inputProbabilities.items():
                self._inputProbability[capsule] = probability

        if inputAttributes is not None:
            for capsule, attributeValueDict in inputAttributes.items():
                self._inputAttributes[capsule] = {}
                for attribute, value in attributeValueDict.items():
                    self._inputAttributes[capsule][attribute] = value

        for attribute, value in outputAttributes.items():
            self._outputAttributes[attribute] = value


    def getOutputs(self):
        return self._outputAttributes    # Attribute - Value
        

    def setOutput(self, attribute : Attribute, value : float):
        if attribute in self._outputAttributes:
            self._outputAttributes[attribute] = value


    def getOutput(self, attribute : Attribute):
        if attribute in self._outputAttributes:
            return self._outputAttributes[attribute]


    def getProbability(self):
        return self._outputProbability


    def getInputProbability(self, capsule):
        if capsule in self._inputProbability:
            return self._inputProbability[capsule]
        else:
            return 0.0


    def getTakenRoute(self):
        return self._route


    def offset(self, offsetLabelX : str, offsetLabelY : str, offsetLabelRatio : str, targetLabelX : str, targetLabelY : str, targetLabelSize : str):
        # We can only offset by one set of attributes, so get any Capsule from the list (Should only be
        # one when this method is called anyways)
        inputCapsule = list(self._inputAttributes.keys())[0]
        offsetX = self._inputAttributes[inputCapsule][inputCapsule.getAttributeByName(offsetLabelX)]
        offsetY = self._inputAttributes[inputCapsule][inputCapsule.getAttributeByName(offsetLabelY)]
        offsetRatio = self._inputAttributes[inputCapsule][inputCapsule.getAttributeByName(offsetLabelRatio)]

        for attribute, value in self._outputAttributes.items():
            if attribute.getName() == targetLabelX:
                self._outputAttributes[attribute] = value * offsetRatio + offsetX
            if attribute.getName() == targetLabelY:
                self._outputAttributes[attribute] = value * offsetRatio + offsetY
            if attribute.getName() == targetLabelSize:
                self._outputAttributes[attribute] = value * offsetRatio
