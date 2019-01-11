
from Capsule import Capsule
from CapsuleMemory import CapsuleMemory
from NeuralNet import NeuralNet
from Attribute import Attribute
from Utility import Utility

class CapsuleRoute:

    def __init__(self, parentCapsule : Capsule, capsuleRouteName : str):
        self._name                  : str           = capsuleRouteName
        self._memory                : CapsuleMemory = CapsuleMemory()
        self._neuralNetGamma        : NeuralNet     = NeuralNet(capsuleRouteName + "-gamma")
        self._neuralNetG            : NeuralNet     = NeuralNet(capsuleRouteName + "-g")
        self._fromCapsules          : list          = list()                        # Capsules
        self._parentCapsule         : Capsule       = parentCapsule

        self._gFunctionLambda                       = None 
        self._gammaFunctionLambda                   = None
        self._gInputMapping         : dict          = dict()  # Attribute - Index 
        self._gOutputMapping        : dict          = dict()  # Index - Attribute
        self._gammaInputMapping     : dict          = dict()  # Attribute - Index
        self._gammaOutputMapping    : dict          = dict()  # Index - Attribute

    def getFromCapsules(self):
        return self._fromCapsules

    def getInputAttributes(self):
        return [x.getAttributes() for x in self._fromCapsules]
        
    def getOutputAttributes(self):
        return self._parentCapsule.getAttributes()

    def resizeInternals(self):
        self._memory.inferXAttributes(self.getInputAttributes())
        self._memory.inferYAttributes(self.getOutputAttributes())

        # TODO: Resize Neural Net

    def runGammaFunction(self):
        outputs : dict = dict()    # Attribute - Value

        # Attributes covered by predefined known \gamma
        missingOutputs = self.getInputAttributes()

        if self._gammaFunctionLambda is not None:
            oldMap : dict = dict()  # Index - Attribute
            for idx, attribute in enumerate(missingOutputs):
                oldMap[idx] = attribute

            gammaInputs = Utility.mapData([x.getValue() for x in missingOutputs], oldMap, self._gammaInputMapping)
            outputs = Utility.mapDataOneWay(self._gammaFunctionLambda(gammaInputs), self._gammaOutputMapping)

        # Attributes covered by general known \gamma
        missingOutputs = [x for x in missingOutputs if x not in outputs.keys()]

        for attribute in missingOutputs:
            value = self.predefinedKnownGammaAttributeFromInputs(attribute)
            if value is not None:
                outputs[attribute] = value


        # Attributes covered by unknown \gamma
        missingOutputs = [x for x in missingOutputs if x not in outputs.keys()]

        # TODO: Run Neural Net on remaining self._neuralNetGamma

        return outputs


    def runGFunction(self):
        return 0


    def predefinedKnownGammaAttributeFromInputs(self, attribute : Attribute):
        # TODO: Mean Calculations and Co.

        # TODO: If NOT covered:
        return None