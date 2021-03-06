
from CapsuleMemory import CapsuleMemory
from NeuralNet import NeuralNet
from NeuralNetGamma import NeuralNetGamma
from NeuralNetG import NeuralNetG
from Attribute import Attribute
from Utility import Utility
from Observation import Observation
from itertools import permutations
from HyperParameters import HyperParameters

import math

class CapsuleRoute:

    def __init__(self, parentCapsule, capsuleRouteName : str, fromCapsules : list):
        self._name                  : str            = capsuleRouteName
        self._memory                : CapsuleMemory  = CapsuleMemory()
        self._fromCapsules          : list           = fromCapsules      # Capsules
        self._parentCapsule                          = parentCapsule

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


    def getJSONMain(self):
        return self._memory.getJSONMain()
 
    
    def getJSONMemory(self):
        return {"route" : self._name, "memory" : self._memory.getJSONMemory()}


    def getName(self):
        return self._name


    def haveSameParent(self, capsules : list):
        # capsules  # List of Capsules
        for caps in capsules:
            if caps not in self._fromCapsules:
                return False

        return True        

    def addSavedObservations(self, observations : list):
        self._memory.addSavedObservations(observations)
        
    def addObservations(self, observations : list):
        self._memory.addObservations(observations)

    def clearObservations(self):
        self._memory.clearObservations()

    def getObservations(self):
        return self._memory.getObservations()

    def getObservation(self, index : int):
        return self._memory.getObservation(index)

    def getNumObservations(self):
        return self._memory.getNumObservations()

    def cleanupObservations(self, offsetLabelX : str, offsetLabelY : str, offsetLabelRatio : str, targetLabelX : str, targetLabelY : str, targetLabelSize : str):
        self._memory.cleanupObservations(lambda attributes: self.applySymmetries(attributes), offsetLabelX, offsetLabelY, offsetLabelRatio, targetLabelX, targetLabelY, targetLabelSize)

    def removeObservation(self, observation : Observation):
        return self._memory.removeObservation(observation)

    def isSemantic(self):
        return self._isSemanticCapsule

    def getInputCapsuleCount(self):
        counts = {}
        for caps in self._fromCapsules:
            if caps in counts:
                counts[caps] = counts[caps] + 1
            else:
                counts[caps] = 1

        return counts # Capsule - Count

    def getProbabilityCutOff(self):
        if self._isSemanticCapsule is True:
            return HyperParameters.SemanticProbabilityCutOff
        else:
            return HyperParameters.PrimitiveProbabilityCutOff


    def getMeanProbability(self):
        return self._memory.getMeanProbability()


    def observationFromInputs(self, inputObservations : list, forcedProbability : float = 1.0):
        inputs = {}                             # Attribute - List of Values
        for obs in inputObservations:
            newInputs = obs.getOutputs(True)    # Attribute - Value
            for newAttr, newValue in newInputs.items():
                if newAttr in inputs:
                    inputs[newAttr].append(newValue)
                else:
                    inputs[newAttr] = [newValue]

        outputs = self.runGammaFunction(inputs, False)

        # TODO: Use actual probability
        return Observation(self._parentCapsule, self, inputObservations, outputs, min(forcedProbability, 1.0))


    def getAttributeDistance(self, fromObservations : list, attribute : Attribute, attributeValue : float):
        # This is an trivial implementation to find an initial guess for the "distance" between
        # Attributes, without knowledge of the geometry of the configuration space.
        # For a more accurate estimate, a bayesian approach should be chosen to incorporate 
        # gained knowledge of the geometry.
        # TODO: This is highly unoptimized...
        newObs = self.observationFromInputs(fromObservations, 1.0)
        targetAttr = newObs.getOutputsList()
        targetAttr[attribute] = [0.0]

        zeroPoint = self.runGFunction(targetAttr, isTraining = False)
        
        targetAttr[attribute] = [attributeValue]

        offPoint = self.runGFunction(targetAttr, isTraining = False)

        totalLen = 0.0
        for i in range(len(zeroPoint)):
            totalLen += (offPoint[i] - zeroPoint[i]) * (offPoint[i] - zeroPoint[i])

        return totalLen / float(len(zeroPoint))


    def getAttributeDistanceRaw(self, fromObservations : list, attribute : Attribute):
        # See self.getAttributeDistance()
        newObs = self.observationFromInputs(fromObservations, 1.0)
        targetAttr = newObs.getOutputsList()
        targetAttr[attribute] = [0.0]

        zeroPoint = self.runGFunction(targetAttr, isTraining = False)
        
        offPoint = Utility.mapDataOneWayDictRevList(newObs.getInputs(), self._gammaInputMapping)

        totalLen = 0.0
        for i in range(len(zeroPoint)):
            totalLen += (offPoint[i] - zeroPoint[i]) * (offPoint[i] - zeroPoint[i])

        return totalLen / float(len(zeroPoint))


    def createSemanticRoute(self, initialObservations : list):
        self._isSemanticCapsule = True

        self.addTrainingData(initialObservations, 1.0)
        self._memory.setLambdaKnownGamma((lambda attributes : self.runGammaFunction(attributes)))
                
        self.resizeInternals()          


    def addTrainingData(self, observations : list, forcedProbability : float = 1.0, appendAttr : Attribute = None, appendValue : float = 0.0):
        newObs = self.observationFromInputs(observations, forcedProbability)

        if appendAttr is not None:
            newObs.appendOutputAttribute(appendAttr, appendValue)

        self._memory.addSavedObservations([newObs])


    def rescaleAttribute(self, attribute : Attribute, scale : float):
        self._memory.rescaleAttribute(attribute, scale)


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
  

    def pairInputCapsuleAttributes(self, attributes : dict):
        # attributes        # Attribute - List of Values
        # We mostly work with {Attribute - List of Values} dictionaries without the associated capsule
        # Here we just pair them back up again. Used to store in Observation
        outputs = {}
        for inputCapsule in self._fromCapsules:
            outputs[inputCapsule] = {}
            for attribute, valueList in attributes.items():
                if inputCapsule.hasAttribute(attribute):
                    outputs[inputCapsule][attribute] = valueList
        return outputs  # Capsule - {Attribute - List of Values}



    def resizeInternals(self):
        prevSizeGamma = 0
        prevSizeG = 0

        if self._gInputMapping is not None:
            prevSizeG = len(self._gInputMapping)
        if self._gammaInputMapping is not None:
            prevSizeGamma = len(self._gammaInputMapping)

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

        hasChanged = False
        if prevSizeG > 0 and prevSizeGamma > 0 and (prevSizeG != len(self._gInputMapping) or prevSizeGamma != len(self._gammaInputMapping)):
            hasChanged = True

        if self._neuralNetG.hasTraining() is True and hasChanged is True:
            self._neuralNetG.delete()

        if self._neuralNetG.hasTraining() is False or hasChanged is True:
            self.retrain()


    def retrain(self, showDebugOutput : bool = True, specificSplit : list = None, fromScratch : bool = False):
        if self._isSemanticCapsule is True:
            if fromScratch is True and self._neuralNetG.hasTraining() is True:
                self._neuralNetG.delete()
            self._neuralNetG.trainFromData(self._memory, showDebugOutput, specificSplit)
        else:
            if fromScratch is True and self._neuralNetGamma.hasTraining() is True:
                self._neuralNetGamma.delete()
            self._neuralNetGamma.trainFromData(self._memory, showDebugOutput, specificSplit)



    def getSymmetry(self, attributes : dict):
        # attributes        # Attribute - List of Values

        # Find symmetry from outputs

        if self._isSemanticCapsule is True and self._neuralNetG is None:
            return attributes

        # We try to find the symmetries on the fly, as they are
        # highly dependend on the current attributes
        highestAgreementN = 0

        copyRotations = {}  # Attribute - List of Values

        for attr, valueList in attributes.items():
            copyRotations[attr] = valueList.copy()
            
        originalResult = self.runGFunction(attributes, isTraining = False)

        n = 2
        while(n <= 20):    
            gResult = {}
            agreement = {}
            testAngle = 1.0 / float(n)

            for attr, valueList in copyRotations.items():
                if attr.getName() in self._rotationalLabels:
                    for idx in range(len(attributes[attr])):
                        copyRotations[attr][idx] = (attributes[attr][idx] + testAngle) % 1.0

            gResult = self.runGFunction(copyRotations, isTraining = False)
            agreement = self.agreementFunction(originalResult, gResult)

            agreementSum = 0.0
            totLen = 0
            for attr, valueList in agreement.items():
                agreementSum += sum(valueList)
                totLen += len(valueList)
            agreementSum = agreementSum / max(1, totLen)

            # TODO: This is only for 1 Axis! 
            if agreementSum > HyperParameters.SymmetryCutOff:
                # Yes, we do have a Symmetry! Can we go deeper?
                highestAgreementN = n
                n = n * 2
            elif highestAgreementN > 0 or (highestAgreementN == 0 and n >= 9):
                # Either we found symmetry
                # or we are doing so tiny rotations that agreement will happen by default
                break
            else:
                n = n + 1
        
        return (1 / max(1, highestAgreementN))


    def getSymmetryInverse(self, attributes : dict):
        # attributes        # Attribute - List of Values

        # Find symmetry from inputs

        # We try to find the symmetries on the fly, as they are
        # highly dependend on the current attributes
        highestAgreementN = 0


        originalResult = self.runGammaFunction(attributes, isTraining = False)

        # TODO: Remove all references to const strings

        centerX = [valueList for (key, valueList) in originalResult.items() if key.getName() == "Position-X"][0][0]
        centerY = [valueList for (key, valueList) in originalResult.items() if key.getName() == "Position-Y"][0][0]


        originalInputs = {}  # Attribute - List of Values
        copyInputs = {}  # Attribute - List of Values

        for attr, valueList in attributes.items():
            if attr.getName() == "Position-X":
                originalInputs[attr] = []
                for val in valueList:
                    originalInputs[attr].append(val - centerX)
            elif attr.getName() == "Position-Y":
                originalInputs[attr] = []
                for val in valueList:
                    originalInputs[attr].append(val - centerY)
            else:
                originalInputs[attr] = valueList.copy()
                

        for attr, valueList in originalInputs.items():
            copyInputs[attr] = valueList.copy()
                  
        
        n = 2
        while(n <= 20):    
            gResult = {}
            agreement = {}
            testAngle = 1.0 / float(n)

            for caps in self._fromCapsules:
                xAttr = caps.getAttributeByName("Position-X")
                yAttr = caps.getAttributeByName("Position-Y")
                rAttr = caps.getAttributeByName("Rotation")

                for idx in range(len(copyInputs[xAttr])):
                    copyInputs[xAttr][idx] = originalInputs[xAttr][idx] * math.cos(-testAngle * math.pi * 2.0) - originalInputs[yAttr][idx] * math.sin(-testAngle * math.pi * 2.0)
                    copyInputs[yAttr][idx] = originalInputs[xAttr][idx] * math.sin(-testAngle * math.pi * 2.0) + originalInputs[yAttr][idx] * math.cos(-testAngle * math.pi * 2.0)
                    # TODO: Apply local rotational symmetries
                    # copyInputs[rAttr][idx] = originalInputs[rAttr][idx] + testAngle


            agreement = self.agreementFunction(copyInputs, originalInputs)

            agreementSum = 0.0
            totLen = 0
            for attr, valueList in agreement.items():
                # TODO: Only for testing till local symmetries are in
                if attr.getName() != "Rotation":
                    agreementSum += sum(valueList)
                    totLen += len(valueList)
            agreementSum = agreementSum / totLen
            
            # TODO: This is only for 1 Axis! 
            if agreementSum > HyperParameters.SymmetryCutOff:
                # Yes, we do have a Symmetry! Can we go deeper?
                highestAgreementN = n
                n = n * 2
            elif highestAgreementN > 0 or (highestAgreementN == 0 and n >= 9):
                # Either we found symmetry
                # or we are doing so tiny rotations that agreement will happen by default
                break
            else:
                n = n + 1
                
        return (1 / max(1, highestAgreementN))


    def applySymmetries(self, attributes : dict):
        # attributes        # Attribute - List of Values

        symmetry = self.getSymmetry(attributes)
        
        for attr in attributes.keys():
            if attr.getName() in self._rotationalLabels:
                for idx in range(len(attributes[attr])):
                    attributes[attr][idx] = attributes[attr][idx] % symmetry

        return attributes   # Attribute - List of Values


    def runGammaFunction(self, attributes : dict = None, isTraining : bool = True):
        # attributes        # Attribute - List of Values

        if self._isSemanticCapsule is False:
            return self._neuralNetGamma.forwardPass(attributes) # Attribute - List of Values
        else:
            # TODO: ACTUAL Semantic Calculation
            outputs = {}
            for attribute in self.getOutputAttributes():
                count = 0
                aggregate = 0.0
                for inAttr, inValueList in attributes.items():
                    if inAttr.getName() == attribute.getName():
                        # TEST: 
                        #count = 1
                        #aggregate = inValueList[0]
                        #break
                        count = count + len(inValueList)
                        aggregate = aggregate + sum(inValueList)
                outputs[attribute] = [aggregate / max(count, 1)]
            
            return outputs  # Attribute - List of Values


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
            outputs = self.semanticAgreementFunction(attributes1, attributes2)
            
        return outputs # Attribute - List of Value


    def semanticAgreementFunction(self, attributes1 : dict, attributes2 : dict):
        # attributes1       # Attribute - List of Values
        # attributes2       # Attribute - List of Values
        outputs = {}
        bestAgreement = 0.0
        for caps, count in self.getInputCapsuleCount().items():
            newOutputs = {}
            for capsPerm in permutations(range(count)):
                testOutputs = {}
                # TODO: Weigh each attribute by importance to agreement
                for attr in caps.getAttributes():
                    testOutputs[attr] = []
                    for idx1, idx2 in enumerate(capsPerm):
                        testOutputs[attr].append(Utility.windowFunction(attributes1[attr][idx1] - attributes2[attr][idx2], HyperParameters.SemAgreementWidth, HyperParameters.SemAgreementFallOff))

                testAgreement = sum([sum(valList) for valList in testOutputs.values()])
                if testAgreement > bestAgreement:
                    newOutputs = testOutputs
            outputs.update(newOutputs)

        return outputs # Attribute - List of Value
        