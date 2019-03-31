from Attribute import Attribute

class Observation:

    def __init__(self, capsule, route, inputObservations : list, outputAttributes : dict, outputProbability : float):
        # inputObservations                                                 # List of Observations      or   List of Lists of Observations
        # outputAttributes                                                  # Attribute - Value         or   Attribute - List of Values
        # inputProbabilities                                                # Capsule - Probability
        self._inputObservations     : list          = []                    # Observations
        self._outputAttributes      : dict          = {}                    # Attribute - Value
        self._outputProbability     : float         = outputProbability
        self._route                                 = route
        self._capsule                               = capsule

        for obs in inputObservations:
            if type(obs) is list:
                for actualObs in obs:
                    self._inputObservations.append(actualObs)
            else:
                self._inputObservations.append(obs)


        for attribute, value in outputAttributes.items():
            if type(value) is list:
                self._outputAttributes[attribute] = value[0]
            else:
                self._outputAttributes[attribute] = value


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
        print("Probability - " + str(int(self._outputProbability * 100)) + "%")
        for attribute, value in self._outputAttributes.items():
            if includeNonInheritable is True or attribute.isInheritable() is True:
                print(attribute.getName() + ": " + str(value))


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