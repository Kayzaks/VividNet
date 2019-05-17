from Attribute import Attribute
from HyperParameters import HyperParameters

class Observation:

    def __init__(self, capsule, route, inputObservations : list, outputAttributes : dict, outputProbability : float, attrIndex : int = 0):
        # inputObservations                                                 # List of Observations      or   List of Lists of Observations
        # outputAttributes                                                  # Attribute - Value         or   Attribute - List of Values
        # inputProbabilities                                                # Capsule - Probability
        self._inputObservations     : list          = []                    # Observations
        self._outputAttributes      : dict          = {}                    # Attribute - Value
        self._outputProbability     : float         = outputProbability
        self._route                                 = route
        self._capsule                               = capsule
        self._previousObservation                   = None
        self._accelerations         : dict          = {}                    # Attribute - Value

        for obs in inputObservations:
            if type(obs) is list:
                for actualObs in obs:
                    self._inputObservations.append(actualObs)
            else:
                self._inputObservations.append(obs)


        for attribute, value in outputAttributes.items():
            if type(value) is list:
                self._outputAttributes[attribute] = value[attrIndex]
            else:
                self._outputAttributes[attribute] = value


    def isParent(self, observation):
        for obs in self._inputObservations:
            if obs == observation:
                return True
        return False


    def getOutputs(self, onlyInheritable : bool = False):
        if onlyInheritable is False:
            return self._outputAttributes    # Attribute - Value
        else:
            outputDict = {}
            for attribute, value in self._outputAttributes.items():
                if attribute.isInheritable() is True:
                    outputDict[attribute] = value
            return outputDict               # Attribute - Value
            

    def getOutputsList(self, onlyInheritable : bool = False):
        outputDict = {}
        for attribute, value in self._outputAttributes.items():
            if onlyInheritable is False or attribute.isInheritable() is True:
                outputDict[attribute] = [value]
        return outputDict               # Attribute -  List of Values
        

    def setOutput(self, attribute : Attribute, value : float):
        if attribute in self._outputAttributes:
            self._outputAttributes[attribute] = value


    def getInputs(self):
        inputs = {}     # Attribute - List of values
        for obs in self._inputObservations:
            newInputs = obs.getOutputs()
            for attr, value in newInputs.items():
                if attr in inputs:
                    inputs[attr].append(value)
                else:
                    inputs[attr] = [value]
        return inputs   # Attribute - List of values


    def getOutput(self, attribute : Attribute = None, attributeName : str = None):
        if attribute is not None and attribute in self._outputAttributes:
            return self._outputAttributes[attribute]
        if attributeName is not None:
            for attr, value in self._outputAttributes.items():
                if attr.getName() == attributeName:
                    return value
        return 0.0

    
    def printOutputs(self, includeNonInheritable : bool):
        print(self._capsule.getName())
        print("Probability - " + str(int(self._outputProbability * 100)) + "%")
        for attribute, value in self._outputAttributes.items():
            if includeNonInheritable is True or attribute.isInheritable() is True:
                print(attribute.getName() + ": " + str(value))
        print("-------------------------")


    def getProbability(self):
        return self._outputProbability


    def getInputProbability(self, capsule):
        for obs in self._inputObservations:
            if obs.getCapsule() == capsule:
                return obs.getProbability()
        return 0.0


    def getTakenRoute(self):
        return self._route


    def getInputObservations(self):
        return self._inputObservations


    def addInputObservation(self, observation):
        self._inputObservations.append(observation)


    def getCapsule(self):
        return self._capsule


    def isZeroObservation(self):
        for value in self._outputAttributes.values():
            if abs(value) > 0.01:
                return False
        return True

    
    def offset(self, offsetLabelX : str, offsetLabelY : str, offsetLabelRatio : str, targetLabelX : str, targetLabelY : str, targetLabelSize : str):
        # We can only offset by one set of attributes, so get any Capsule from the list (Should only be
        # one when this method is called anyways)
        inputCapsule = self._inputObservations[0].getCapsule()
        offsetX = self._inputObservations[0].getOutputs()[inputCapsule.getAttributeByName(offsetLabelX)]
        offsetY = self._inputObservations[0].getOutputs()[inputCapsule.getAttributeByName(offsetLabelY)]
        offsetRatio = self._inputObservations[0].getOutputs()[inputCapsule.getAttributeByName(offsetLabelRatio)]

        for attribute, value in self._outputAttributes.items():
            if attribute.getName() == targetLabelX:
                self._outputAttributes[attribute] = value * offsetRatio + offsetX
            if attribute.getName() == targetLabelY:
                self._outputAttributes[attribute] = value * offsetRatio + offsetY
            if attribute.getName() == targetLabelSize:
                self._outputAttributes[attribute] = value * offsetRatio


    def cleanupSymmetries(self, applySymmetries):
        # applySymmetries  # Lambda attributes -> attributes
        newOutputs = applySymmetries(self.getOutputsList())

        for attr, valList in newOutputs.items():
            self._outputAttributes[attr] = valList[0]


    def linkPreviousObservation(self, observation):
        self._previousObservation = observation


    def setAccelerations(self, accelerations : dict):
        self._accelerations = accelerations


    def getVelocities(self, timeStep : float):
        velocities = {}
        if self._previousObservation is None:
            for attr, value in self._outputAttributes.items():
                velocities[attr] = 0.0
        else:            
            linkedOutputs = self._previousObservation.getOutputs()
            for attr, value in self._outputAttributes.items():
                velocities[attr] = (value - linkedOutputs[attr]) / timeStep
                if attr in self._accelerations:
                    velocities[attr] = velocities[attr] + 0.5 * self._accelerations[attr] * timeStep
                if velocities[attr] < HyperParameters.VelocityCutoff:
                    # "Smooth" out small error fluctuations
                    velocities[attr] = 0.0
        return velocities   # {Attribute, Velocity}

    

