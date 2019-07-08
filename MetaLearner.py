
from Utility import Utility

import numpy as np

class MetaLearner:
    def __init__(self):
        self._decisionMatrix = {}           # Feature Lambda - List of Entries
                                            # [A.1, A.2, B.1, B.2]
        self._lambdaID       = {}           # Index - Feature Lambda
        self._lastFeatures   = []           # List of Feature Lambdas


    def getJSON(self):
        outData = {}
        for idKey, feature in self._lambdaID.items():
            outData[idKey] = self._decisionMatrix[feature]

        return outData


    def putJSON(self, data):
        for idStr, valStrList in data.items():
            idKey = Utility.safeCast(idStr, int, -1)

            if idKey in self._lambdaID and len(valStrList) == 4:
                valList = []
                for i in range(4):
                    valList.append(Utility.safeCast(valStrList[i], int, 0))
                self._decisionMatrix[self._lambdaID[idKey]] = valList


    def addLambda(self, featureLambda):
        # featureLambda  # Input  : {Capsule, List of Observations}, {Observed Axioms (Capsule), List of Observations}
        #                # Output : Boolean
        self._decisionMatrix[featureLambda] = np.array([0, 0, 0, 0])
        self._lambdaID[len(self._lambdaID)] = featureLambda


    def checkResults(self, observations : dict, observedAxioms : dict):
        totalDecision = np.array([0, 0, 0, 0])
        self._lastFeatures = []

        for featureLambda in self._decisionMatrix.keys():
            if featureLambda(observations, observedAxioms) is True:
                totalDecision = np.add(totalDecision, self._decisionMatrix[featureLambda])
                self._lastFeatures.append(featureLambda)

        maxEntry = np.argmax(totalDecision)

        if maxEntry == 0:
            return "A non-activated parent capsule is missing a route."
        elif maxEntry == 1:
            return "A parent capsule is missing."
        elif maxEntry == 2:
            return "An attribute is lacking training data."
        elif maxEntry == 3:
            return "An attribute is missing."


    def applyOracle(self, oracleDecision : int):
        # oracleDecision    # Trigger Index

        for featureLambda in self._lastFeatures:
            self._decisionMatrix[featureLambda][oracleDecision] += 1