from Attribute import Attribute
from CapsuleRoute import CapsuleRoute

class Observation:

    def __init__(self, route : CapsuleRoute, inputAttributes : dict, outputAttributes : dict):
        # inputAttributes                                    # Capsule - {Attribute - Value}
        # outputAttributes                                   # Attribute - Value
        self._inputProbability      : dict          = {}     # Capsule - Probability
        self._inputAttributes       : dict          = {}     # Capsule - {Attribute - Value}
        self._outputAttributes      : dict          = {}     # Attribute - Value
        self._route                 : CapsuleRoute  = route

        for capsule, attributeValueDict in inputAttributes.items():
            self._inputAttributes[capsule] = {}
            for attribute, value in attributeValueDict.items():
                self._inputAttributes[capsule][Attribute] = value

        for attribute, value in outputAttributes.items():
            self._outputAttributes[attribute] = value


    def getOutputs(self):
        return self._outputAttributes    # Attribute - Value


    def offset(self, offsetLabelX : str, offsetLabelY : str, targetLabelX : str, targetLabelY : str):
        # We can only offset by one set of attributes, so get any Capsule from the list (Should only be
        # one when this method is called anyways)
        inputCapsule = self._inputAttributes.keys()[0]
        offsetX = self._inputAttributes[inputCapsule][inputCapsule.getAttributeByName(offsetLabelX)]
        offsetY = self._inputAttributes[inputCapsule][inputCapsule.getAttributeByName(offsetLabelY)]

        for attribute, value in self._outputAttributes.items():
            if attribute.getName() == targetLabelX:
                self._outputAttributes[attribute] = value + offsetX
            if attribute.getName() == targetLabelY:
                self._outputAttributes[attribute] = value + offsetY