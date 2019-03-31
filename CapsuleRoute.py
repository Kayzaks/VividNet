
from CapsuleMemory import CapsuleMemory
from NeuralNet import NeuralNet
from NeuralNetGamma import NeuralNetGamma
from NeuralNetG import NeuralNetG
from Attribute import Attribute
from Utility import Utility
from Observation import Observation

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

        self._gInputMapping         : dict           = None  # Attribute - List of Indices (Always 1 Element)
        self._gOutputMapping        : dict           = None  # Index - Attribute
        self._gammaInputMapping     : dict           = None  # Attribute - List of Indices
        self._gammaOutputMapping    : dict           = None  # Index - Attribute
        self._neuralNetGamma        : NeuralNetGamma = None
        self._neuralNetG            : NeuralNetG     = None

        self._isSemanticCapsule     : bool           = True
        # TODO: Get Rotational Label from PrimitivesRenderer
        self._rotationalLabels      : list           = ["Rotation"] # Per Axis


    def addObservations(self, observation : Observation):
        self._memory.addObservations(observation)

    def clearObservations(self):
        self._memory.clearObservations()

    def getObservations(self):
        return self._memory.getObservations()

    def getObservation(self, index : int):
        return self._memory.getObservation(index)

    def getNumObservations(self):
        return self._memory.getNumObservations()

    def cleanupObservations(self, offsetLabelX : str, offsetLabelY : str, offsetLabelRatio : str, targetLabelX : str, targetLabelY : str, targetLabelSize : str):
        self._memory.cleanupObservations(offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize)

    def removeObservation(self, observation : Observation):
        return self._memory.removeObservation(observation)

    def isSemantic(self):
        return self._isSemanticCapsule


    def createSemanticRoute(self, initialObservations : list):
        self._isSemanticCapsule = True

        inputs = {}                             # Attribute - List of Values
        for obs in initialObservations:
            newInputs = obs.getOutputs(True)    # Attribute - Value
            for newAttr, newValue in newInputs.items():
                if newAttr in inputs:
                    inputs[newAttr].append(newValue)
                else:
                    inputs[newAttr] = [newValue]

        outputs = self.runGammaFunction(inputs)
        newObservation = Observation(self._parentCapsule, self, initialObservations, outputs, 1.0)

        self._memory.addSavedObservations([newObservation])
        self._memory.setLambdaKnownGamma((lambda attributes : self.runGammaFunction(attributes)))
                
        self.resizeInternals()            


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
        return [item for sublist in [x.getAttributes() for x in self._fromCapsules] for item in sublist if item.isInheritable() is True]
                
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
        # TODO: Check if anything actually changed
        self._gInputMapping         : dict          = dict()  # Attribute - List of Indices (with 1 Element)
        self._gOutputMapping        : dict          = dict()  # Index - Attribute
        self._gammaInputMapping     : dict          = dict()  # Attribute - List of Indices
        self._gammaOutputMapping    : dict          = dict()  # Index - Attribute

        for idx, attribute in enumerate(self.getInputAttributes()):
            if attribute in self._gammaInputMapping:
                self._gammaInputMapping[attribute].append(idx)
            else:
                self._gammaInputMapping[attribute] = [idx]
            self._gOutputMapping[idx] = attribute
            
        for idx, attribute in enumerate(self.getOutputAttributes()):
            self._gInputMapping[attribute] = [idx]
            self._gammaOutputMapping[idx] = attribute

        self._neuralNetG            : NeuralNetG     = NeuralNetG(self._gInputMapping, self._gammaInputMapping, self._name + "-g", True)

        if self._neuralNetG.hasTraining() is True:
            xx = 0
            # TODO: Delete Previous Training File

        # TODO: Only Retrain if changed!
        if self._neuralNetG.hasTraining() is False:
            self.retrain()


    def retrain(self, showDebugOutput = True, specificSplit : list = None):
        if self._isSemanticCapsule is True:
            self._neuralNetG.trainFromData(self._memory, showDebugOutput, specificSplit)
        else:
            self._neuralNetGamma.trainFromData(self._memory, showDebugOutput, specificSplit)


    def applySymmetries(self, attributes : dict):
        # attributes        # Attribute - List of Values

        # We try to find the symmetries on the fly, as they are
        # highly dependend on the current attributes
        highestAgreementN = 1
        highestAgreement = 0.9

        originalRotations = {}  # Attribute - List of Values

        for attr, valueList in attributes.items():
            if attr.getName() in self._rotationalLabels:
                originalRotations[attr] = valueList.copy()
            
        originalResult = self.runGFunction(attributes, isTraining = False)

        for n in range(2, 10):
            testAngle = 1.0 / n

            for attr, valueList in attributes.items():
                if attr.getName() in self._rotationalLabels:
                    for idx in range(len(originalRotations[attr])):
                        valueList[idx] = (originalRotations[attr][idx] + testAngle) % 1.0

            gResult = self.runGFunction(attributes, isTraining = False)
            agreement = self.agreementFunction(originalResult, gResult)

            agreementSum = 0.0
            for attr, value in agreement.items():
                agreementSum += value
            agreementSum = agreementSum / len(agreement)

            # TODO: This is only for 1 Axis! 
            if highestAgreement - agreementSum < 0.001:
                highestAgreement = agreementSum
                highestAgreementN = n
        
        for attr, valueList in attributes.items():
            if attr.getName() in self._rotationalLabels:
                for idx in range(len(attributes[attr])):
                    attributes[attr][idx] = originalRotations[attr][idx] % (1 / highestAgreementN)

        return attributes   # Attribute - List of Values


    def runGammaFunction(self, attributes : dict = None):
        # attributes        # Attribute - List of Values

        if self._isSemanticCapsule is False:
            return self.applySymmetries(self._neuralNetGamma.forwardPass(attributes)) # Attribute - List of Values
        else:
            # TODO: ACTUAL Semantic Calculation
            outputs = {}
            for attribute in self.getOutputAttributes():
                count = 0
                aggregate = 0.0
                for inAttr, inValueList in attributes.items():
                    if inAttr.getName() == attribute.getName():
                        count = count + len(inValueList)
                        aggregate = aggregate + sum(inValueList)

                outputs[attribute] = [aggregate / max(count, 1)]
            return self.applySymmetries(outputs)  # Attribute - List of Values


    def runGFunction(self, attributes : dict = None, isTraining : bool = True):
        # attributes        # Attribute - List of Values
        if self._isSemanticCapsule is False:
            values = self._memory.runXInferer(Utility.mapDataOneWayDictRevList(attributes, self._gInputMapping), isTraining)
            return Utility.mapDataOneWayList(values, self._gOutputMapping)   # Attribute - List of Values
        else:
            return self._neuralNetG.forwardPass(attributes)                  # Attribute - List of Values


    def agreementFunction(self, attributes1 : dict, attributes2 : dict):
        # attributes1       # Attribute - List of Values
        # attributes2       # Attribute - List of Values
        outputs = {}
        if self._isSemanticCapsule is False:
            outputs = self._agreementFunctionLambda(attributes1, attributes2)
        else:
            outputs = CapsuleRoute.semanticAgreementFunction(attributes1, attributes2)
            
        return outputs # Attribute - Value


    @staticmethod
    def semanticAgreementFunction(attributes1 : dict, attributes2 : dict):
        # attributes1       # Attribute - List of Values
        # attributes2       # Attribute - List of Values
        outputs = {}
        for attribute, valueList in attributes1.items():
            # TODO: Try out all permutations?
            # TODO: This is wrong, the sum is a bad measure
            outputs[attribute] = Utility.windowFunction(sum(valueList) - sum(attributes2[attribute]), 0.1, 0.1)
            
        return outputs # Attribute - Value
        