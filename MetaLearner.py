
import numpy as np

class MetaLearner:
    def __init__(self):
        self._decisionMatrix = {}           # Feature Lambda - List of Entries
                                            # [A.1, A.2, B.1, B.2]
        self._lastFeatures   = []           # List of Feature Lambdas


    def addLambda(self, featureLambda):
        # featureLambda  # Input: {Capsule, List of Observations}  # Output : Boolean
        self._decisionMatrix[featureLambda] = np.array([0, 0, 0, 0])


    def checkResults(self, observations : list):
        totalDecision = np.array([0, 0, 0, 0])
        self._lastFeatures = []

        for featureLambda in self._decisionMatrix.keys():
            if featureLambda(observations) is True:
                totalDecision = np.add(totalDecision, self._decisionMatrix[featureLambda])
                self._lastFeatures.append(featureLambda)

        maxEntry = np.argmax(totalDecision)

        if maxEntry == 0:
            print("A non-activated parent capsule is missing a route.")
        elif maxEntry == 1:
            print("A parent capsule is missing.")
        elif maxEntry == 2:
            print("An attribute is lacking training data.")
        elif maxEntry == 3:
            print("An attribute is missing.")


    def applyOracle(self, oracleDecision : int):
        # oracleDecision    # Trigger Index

        for featureLambda in self._lastFeatures:
            self._decisionMatrix[featureLambda][oracleDecision] += 1