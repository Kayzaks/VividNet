
from Capsule import Capsule
from CapsuleMemory import CapsuleMemory
from NeuralNet import NeuralNet
from Attribute import Attribute
from Utility import Utility

class CapsuleRoute:

    def __init__(self, parentCapsule : Capsule, capsuleRouteName : str, fromCapsules : list):
        self._name                  : str           = capsuleRouteName
        self._memory                : CapsuleMemory = CapsuleMemory()
        self._fromCapsules          : list          = fromCapsules      # Capsules
        self._parentCapsule         : Capsule       = parentCapsule

        self._gFunctionLambda                       = None 
        self._gammaFunctionLambda                   = None

        self._gInputMapping         : dict          = None  # Attribute - Index 
        self._gOutputMapping        : dict          = None  # Index - Attribute
        self._gammaInputMapping     : dict          = None  # Attribute - Index
        self._gammaOutputMapping    : dict          = None  # Index - Attribute
        self._neuralNetGamma        : NeuralNet     = None
        self._neuralNetG            : NeuralNet     = None

        self.resizeInternals()


    def getFromCapsules(self):
        return self._fromCapsules

    def getInputAttributes(self):
        return [item for sublist in [x.getAttributes() for x in self._fromCapsules] for item in sublist]
        
    def getOutputAttributes(self):
        return self._parentCapsule.getAttributes()

    def getInputActivations(self):
        # TODO: Output as dict(Capsule, Probability)
        return {(self._fromCapsules[0], 1.0)}


    def resizeInternals(self):
        self._memory.inferXAttributes(self.getInputAttributes())
        self._memory.inferYAttributes(self.getOutputAttributes())

        self._gInputMapping         : dict          = dict()  # Attribute - Index 
        self._gOutputMapping        : dict          = dict()  # Index - Attribute
        self._gammaInputMapping     : dict          = dict()  # Attribute - Index
        self._gammaOutputMapping    : dict          = dict()  # Index - Attribute

        for idx, attribute in enumerate(self.getInputAttributes()):
            self._gammaInputMapping[attribute] = idx
            self._gOutputMapping[idx] = attribute
            
        for idx, attribute in enumerate(self.getOutputAttributes()):
            self._gInputMapping[attribute] = idx
            self._gammaOutputMapping[idx] = attribute

        self._neuralNetGamma        : NeuralNet     = NeuralNet(self._gammaInputMapping, self._gInputMapping, self._name + "-gamma")
        self._neuralNetG            : NeuralNet     = NeuralNet(self._gInputMapping, self._gammaInputMapping, self._name + "-g")

        self.retrain()



    def retrain(self):
        # TODO: IF we got data..
        return


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

        finalResults = self._neuralNetGamma.runBlock(dict((x, x.getValue()) for x in missingOutputs))
        for attribute, value in finalResults:
            if attribute not in outputs:
                # We are this specific to avoid overriding already calculated attributes
                outputs[attribute] = value

        return outputs


    def runGFunction(self):
        return 0


    def predefinedKnownGammaAttributeFromInputs(self, attribute : Attribute):
        # TODO: Mean Calculations and Co.

        # TODO: If NOT covered:
        return None