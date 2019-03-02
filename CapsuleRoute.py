
from CapsuleMemory import CapsuleMemory
from NeuralNet import NeuralNet
from NeuralNetGamma import NeuralNetGamma
from NeuralNetG import NeuralNetG
from Attribute import Attribute
from Utility import Utility

from GraphicsUserInterface import GraphicsUserInterface
class CapsuleRoute:

    def __init__(self, parentCapsule, capsuleRouteName : str, fromCapsules : list):
        self._name                  : str            = capsuleRouteName
        self._memory                : CapsuleMemory  = CapsuleMemory()
        self._fromCapsules          : list           = fromCapsules      # Capsules
        self._parentCapsule                          = parentCapsule

        self._gFunctionLambda                        = None 
        self._gammaFunctionLambda                    = None
        self._agreementFunctionLambda                = None

        self._gInputMapping         : dict           = None  # Attribute - Index 
        self._gOutputMapping        : dict           = None  # Index - Attribute
        self._gammaInputMapping     : dict           = None  # Attribute - Index
        self._gammaOutputMapping    : dict           = None  # Index - Attribute
        self._neuralNetGamma        : NeuralNetGamma = None
        self._neuralNetG            : NeuralNetG     = None

        self._isSemanticCapsule     : bool           = True

        self.resizeInternals()


    def isSemantic(self):
        return self._isSemanticCapsule


    def createSemanticRoute(self):
        self._isSemanticCapsule = True
        return

    def createPrimitiveRoute(self, gInputMapping : dict, gOutputMapping : dict, 
                                   gammaInputMapping : dict, gammaOutputMapping : dict,
                                   lambdaGenerator, lambdaRenderer, lambdaAgreement, 
                                   modelSplit : list, width : int, height : int, depth : int):
        self._isSemanticCapsule = False

        self._gInputMapping = gInputMapping
        self._gOutputMapping = gOutputMapping
        self._gammaInputMapping = gammaInputMapping
        self._gammaOutputMapping = gammaOutputMapping

        self._memory.setLambdaKnownG(lambdaGenerator, lambdaRenderer, gOutputMapping, gammaOutputMapping)
        self._agreementFunctionLambda = lambdaAgreement

        self._neuralNetGamma = NeuralNetGamma(self._gammaInputMapping, self._gInputMapping, self._name + "-gamma", False)
        self._neuralNetGamma.setModelSplit(modelSplit)
        self._neuralNetGamma.setInputShape([width, height, depth])

        if self._neuralNetGamma.hasTraining() is False:
            self.retrain()

    def getInputCount(self):
        return len(self._fromCapsules)

    def getFromCapsules(self):
        return self._fromCapsules

    def getInputAttributes(self):
        return [item for sublist in [x.getAttributes() for x in self._fromCapsules] for item in sublist]
                
    def getOutputAttributes(self):
        return self._parentCapsule.getAttributes()
        
    def getInputActivations(self):
        # TODO: Output as dict(Capsule, Probability)
        return {(self._fromCapsules[0], 1.0)}


    def pairInputCapsuleAttributes(self, attributes : dict):
        # attributes        # Attribute - Value
        # We mostly work with {Attribute - Value} dictionaries without the associated capsule
        # Here we just pair them back up again. Used to store in Observation
        outputs = {}
        for inputCapsule in self._fromCapsules:
            outputs[inputCapsule] = {}
            for attribute, value in attributes.items():
                if inputCapsule.hasAttribute(attribute):
                    outputs[inputCapsule][attribute] = value
        return outputs



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



    def retrain(self, showDebugOutput = True, specificSplit : list = None):
        if self._isSemanticCapsule is True:
            return
        else:
            self._neuralNetGamma.trainFromData(self._memory, showDebugOutput, specificSplit)



    def runGammaFunction(self, attributes : dict = None):
        # attributes        # Attribute - Value
        if self._isSemanticCapsule is False:
            return self._neuralNetGamma.forwardPass(attributes)
        else:
            # TODO: Semantic Calculation
            outputs = {}
            for index, attribute in self._gammaOutputMapping.items():
                outputs[attribute] = 0.0
            return outputs


    def runGFunction(self, attributes : dict = None, isTraining : bool = True):
        # attributes        # Attribute - Value
        if self._isSemanticCapsule is False:
            values = self._memory.runXInferer(Utility.mapDataOneWayDictRev(attributes, self._gInputMapping), isTraining)
            return Utility.mapDataOneWay(values, self._gOutputMapping)  # Attribute - Value
        else:
            return self._neuralNetG.forwardPass(attributes)             # Attribute - Value


    def agreementFunction(self, attributes1 : dict, attributes2 : dict):
        # attributes1       # Attribute - Value
        # attributes2       # Attribute - Value
        outputs = {}
        if self._isSemanticCapsule is False:
            outputs = self._agreementFunctionLambda(attributes1, attributes2)
        else:
            outputs = CapsuleRoute.semanticAgreementFunction(attributes1, attributes2)
            
        return outputs # Attribute - Value


    @staticmethod
    def semanticAgreementFunction(attributes1 : dict, attributes2 : dict):
        # attributes1       # Attribute - Value
        # attributes2       # Attribute - Value
        outputs = {}
        for attribute, value in attributes1.items():
            outputs[attribute] = Utility.windowFunction(value - attributes2[attribute], 0.1, 0.1)
            
        return outputs # Attribute - Value
            

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



#    def gNextBatch(self, batchSize):
#        if self._isSemanticCapsule is False:
#            return self._memory.nextBatch(batchSize, self._gammaInputMapping, self._gInputMapping)
#        else:
#            # TODO: Infer?
#            return None
#
#
#        
#    def gammaNextBatch(self, batchSize):
#        if self._isSemanticCapsule is True:
#            # TODO: 
#            return [[0], [0]]
#        else:
#            # TODO: Infer?
#            return None
#
#
#    def knownGammaAttributeFromInputs(self, attribute : Attribute):
#        # TODO: Mean Calculations and Co.
#
#        # TODO: If NOT covered:
#        return None