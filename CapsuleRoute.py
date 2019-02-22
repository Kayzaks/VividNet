
from CapsuleMemory import CapsuleMemory
from NeuralNet import NeuralNet
from NeuralNetGamma import NeuralNetGamma
from NeuralNetG import NeuralNetG
from Attribute import Attribute
from Utility import Utility

class CapsuleRoute:

    def __init__(self, parentCapsule, capsuleRouteName : str, fromCapsules : list):
        self._name                  : str            = capsuleRouteName
        self._memory                : CapsuleMemory  = CapsuleMemory()
        self._fromCapsules          : list           = fromCapsules      # Capsules
        self._parentCapsule                          = parentCapsule

        self._gFunctionLambda                        = None 
        self._gammaFunctionLambda                    = None

        self._gInputMapping         : dict           = None  # Attribute - Index 
        self._gOutputMapping        : dict           = None  # Index - Attribute
        self._gammaInputMapping     : dict           = None  # Attribute - Index
        self._gammaOutputMapping    : dict           = None  # Index - Attribute
        self._neuralNetGamma        : NeuralNetGamma = None
        self._neuralNetG            : NeuralNetG     = None

        self._isSemanticCapsule     : bool           = True

        self.resizeInternals()


    def createSemanticRoute(self):
        self._isSemanticCapsule = True
        return

    def createPrimitiveRoute(self, gInputMapping : dict, gOutputMapping : dict, 
                                   gammaInputMapping : dict, gammaOutputMapping : dict,
                                   lambdaGenerator, lambdaRenderer, modelSplit : list,
                                   width : int, height : int, depth : int):
        self._isSemanticCapsule = False

        self._gInputMapping = gInputMapping
        self._gOutputMapping = gOutputMapping
        self._gammaInputMapping = gammaInputMapping
        self._gammaOutputMapping = gammaOutputMapping

        self._memory.setLambdaKnownG(lambdaGenerator, lambdaRenderer, gOutputMapping, gammaOutputMapping)

        self._neuralNetGamma = NeuralNetGamma(self._gammaInputMapping, self._gInputMapping, self._name + "-gamma", False)
        self._neuralNetGamma.setModelSplit(modelSplit)
        self._neuralNetGamma.setInputShape([width, height, depth])

        if self._neuralNetGamma.hasTraining() is False:
            self.retrain()


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
        if self._isSemanticCapsule is True:
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

            self._neuralNetG            : NeuralNetG     = NeuralNetG(self._gInputMapping, self._gammaInputMapping, self._name + "-g", True)

            self.retrain()



    def retrain(self, specificSplit : list = None):
        if self._isSemanticCapsule is True:
            return
        else:
            self._neuralNetGamma.trainFromData(self._memory, False, specificSplit)


#    def runGammaFunction(self):
#        # TODO: Redesign?
#        outputs : dict = dict()    # Attribute - Value
#
#        # Attributes covered by predefined known \gamma
#        missingOutputs = self.getInputAttributes()
#
#        if self._gammaFunctionLambda is not None:
#            oldMap : dict = dict()  # Index - Attribute
#            for idx, attribute in enumerate(missingOutputs):
#                oldMap[idx] = attribute
#
#            gammaInputs = Utility.mapData([x.getValue() for x in missingOutputs], oldMap, self._gammaInputMapping)
#            outputs = Utility.mapDataOneWay(self._gammaFunctionLambda(gammaInputs), self._gammaOutputMapping)
#
#        # Attributes covered by general known \gamma
#        missingOutputs = [x for x in missingOutputs if x not in outputs.keys()]
#
#        for attribute in missingOutputs:
#            value = self.knownGammaAttributeFromInputs(attribute)
#            if value is not None:
#                outputs[attribute] = value
#
#
#        # Attributes covered by unknown \gamma
#        missingOutputs = [x for x in missingOutputs if x not in outputs.keys()]
#
#        finalResults = self._neuralNetGamma.runBlock(dict((x, x.getValue()) for x in missingOutputs))
#        for attribute, value in finalResults:
#            if attribute not in outputs:
#                # We are this specific to avoid overriding already calculated attributes
#                outputs[attribute] = value
#
#        return outputs


    def runGammaFunction(self, attributes : list = None):
        # TODO: THIS IS FOR TESTING USING RAW NON MAPPED ATTRIBUTES
        if self._isSemanticCapsule is False:
            if attributes is None:
                # TODO: Take actual Attributes
                return [0.0] * len(self._gOutputMapping)
            else:
                inputs = Utility.mapDataOneWay(attributes, self._gOutputMapping)
                outputs = self._neuralNetGamma.forwardPass(inputs)
                return Utility.mapDataOneWayDict(outputs, self._gammaOutputMapping)
        else:
            return [0.0] * len(self._gOutputMapping)


    def gNextBatch(self, batchSize):
        if self._isSemanticCapsule is False:
            return self._memory.nextBatch(batchSize, self._gammaInputMapping, self._gInputMapping)
        else:
            # TODO: Infer?
            return None


        
    def gammaNextBatch(self, batchSize):
        if self._isSemanticCapsule is True:
            # TODO: 
            return [[0], [0]]
        else:
            # TODO: Infer?
            return None


    def knownGammaAttributeFromInputs(self, attribute : Attribute):
        # TODO: Mean Calculations and Co.

        # TODO: If NOT covered:
        return None