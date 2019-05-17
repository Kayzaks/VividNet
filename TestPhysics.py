

from PrimitivesPhysics import PrimitivesPhysics
from RelationTriplet import RelationTriplet
from HyperParameters import HyperParameters
from AttributePool import AttributePool

import numpy as np
import math
import random


class TestPhysics(PrimitivesPhysics):

    def init(self):
        self._xPosOffset = self._attributePool.getAttributeOrderByName("Position-X")
        self._yPosOffset = self._attributePool.getAttributeOrderByName("Position-Y")
        self._sizeOffset = self._attributePool.getAttributeOrderByName("Size")
        self._rotOffset = self._attributePool.getAttributeOrderByName("Rotation")
        self._arOffset = self._attributePool.getAttributeOrderByName("Aspect-Ratio")
        self._intOffset = self._attributePool.getAttributeOrderByName("Intensity")
        self._strOffset = self._attributePool.getAttributeOrderByName("Strength")


    def generateRelation(self):
        # Triplet Format:
        # Sender   -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic
        # Receiver -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic 
        # Relation -- Distance | Degrees-Of-Freedom | Sender Normal | Receiver Normal

        # Effect Format:
        # Acceleration Vector | Angle Acceleration Vector

        triplet = [0.0] * RelationTriplet.tripletLength()
        effect  = [0.0] * HyperParameters.DegreesOfFreedom

        ######### TRIPLET
        totalObjectEntries = (HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2)
        
        if random.randint(0, 100) % 2 == 0:
            triplet[2 * totalObjectEntries] = random.random() * HyperParameters.DistanceCutoff
            hasCollision = True
        else:
            triplet[2 * totalObjectEntries] = random.random() + HyperParameters.DistanceCutoff 
            hasCollision = False

        massSizeA = random.random()
        massSizeB = random.random()

        positionA = np.array([random.random(), random.random()])
        differenceVector = np.array([random.random() - 0.5, random.random() - 0.5])
        differenceVector = differenceVector / np.linalg.norm(differenceVector)
        distanceVector = differenceVector * (triplet[2 * totalObjectEntries] + (massSizeA + massSizeB) * 0.5)
        positionB = np.add(positionA, distanceVector)

        # Slightly Off-Center Interactions
        offRotA = random.random() * 2 * math.pi
        offRotB = random.random() * 2 * math.pi

        velocityA = random.random() * np.array([differenceVector[0] * math.cos(offRotA) - differenceVector[1] * math.sin(offRotA), differenceVector[0] * math.cos(offRotA) + differenceVector[1] * math.sin(offRotA)])
        velocityB = random.random() * np.array([-differenceVector[0] * math.cos(offRotB) + differenceVector[1] * math.sin(offRotB), -differenceVector[0] * math.cos(offRotB) - differenceVector[1] * math.sin(offRotB)])

            
        # Filling Sender Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[HyperParameters.MaximumSymbolCount + i] = random.random()
        triplet[HyperParameters.MaximumSymbolCount + self._xPosOffset] = positionA[0]
        triplet[HyperParameters.MaximumSymbolCount + self._yPosOffset] = positionA[1]
        triplet[HyperParameters.MaximumSymbolCount + self._sizeOffset] = massSizeA
        triplet[HyperParameters.MaximumSymbolCount + self._arOffset] = 1.0
        
        # Filling Receiver Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + i] = random.random()
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._xPosOffset] = positionB[0]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._yPosOffset] = positionB[1]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._sizeOffset] = massSizeB
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._arOffset] = 1.0
        
        # Filling Sender Velocity
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + i] = 0.5
        triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._xPosOffset] = (velocityA[0] + 1.0) / 2.0
        triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._yPosOffset] = (velocityA[1] + 1.0) / 2.0
        
        # Filling Receiver Velocity
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[totalObjectEntries + HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + i] = 0.5
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._xPosOffset] = (velocityB[0] + 1.0) / 2.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._yPosOffset] = (velocityB[1] + 1.0) / 2.0

        # TODO:  The following are fixed for testing only Circles with Dynamic, Rigid Collisions:
        # Circle
        triplet[0] = 1.0
        triplet[totalObjectEntries] = 1.0
        # Static / Dynamic
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
        # Rigid / Elastic
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 0.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 0.0
        # Degrees-Of-Freedom
        triplet[2 * totalObjectEntries + 1] = 1.0
        triplet[2 * totalObjectEntries + 2] = 1.0
        triplet[2 * totalObjectEntries + 3] = 1.0
        # Sender Normal
        triplet[2 * totalObjectEntries + 4] = (differenceVector[0] + 1.0) / 2.0
        triplet[2 * totalObjectEntries + 5] = (differenceVector[1] + 1.0) / 2.0
        # Receiver Normal
        triplet[2 * totalObjectEntries + 6] = (-differenceVector[0] + 1.0) / 2.0
        triplet[2 * totalObjectEntries + 7] = (-differenceVector[1] + 1.0) / 2.0


        ######### EFFECT

        # We take the effects to be the sum of a Force F over time delta-t, i.e. Impuls I = F * delta-t
        # Even though balls are close, they are going in different directions
        if not (np.dot(velocityA, differenceVector) < 0 and (np.dot(velocityB, differenceVector) > 0 or np.linalg.norm(velocityB) < np.linalg.norm(velocityA))) and \
           not (np.dot(velocityB, -differenceVector) < 0 and (np.dot(velocityA, -differenceVector) > 0 or np.linalg.norm(velocityA) < np.linalg.norm(velocityB))) and \
           hasCollision == True:

            tempB = np.dot((velocityB - velocityA), distanceVector) / (math.pow(np.linalg.norm(distanceVector), 2.0))
            resultVelocityB = velocityB - (2 * massSizeA / (massSizeA + massSizeB)) * tempB * distanceVector
            resultAccelB = (resultVelocityB - velocityB) / HyperParameters.TimeStep

            # Scaling the Force Vectors
            effect[0] = (resultAccelB[0] / HyperParameters.AccelerationScale + 1.0) / 2.0
            effect[1] = (resultAccelB[1] / HyperParameters.AccelerationScale + 1.0) / 2.0
            effect[2] = 0.5
        else:
            effect[0] = 0.5
            effect[1] = 0.5
            effect[2] = 0.5



        return triplet, effect


    def generateInteraction(self):
        # Aggregate Format:
        # Receiver -- Attributes | Symbol | Velocities | Static/Dynamic | Rigid/Elastic
        # Effects  -- Summed Effect Acceleration Vector | Summed Effect Angle Acceleration Vector
        # External -- External Acceleration Vector | External Angle Acceleration Vector

        # Attributes Format:
        # Receiver -- Attributes | Accelerations

        aggregate  = [0.0] * (HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + HyperParameters.DegreesOfFreedom * 2 + 2) 
        attributes = [0.0] * HyperParameters.MaximumAttributeCount * 2


        ######### AGGREGATE
        # Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            aggregate[i] = random.random()
        aggregate[self._arOffset] = 1.0

        # Symbol
        # TODO: Just a Circle
        aggregate[HyperParameters.MaximumAttributeCount] = 1.0

        # Velocities
        offset = HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount
        for i in range(HyperParameters.MaximumAttributeCount):
            aggregate[offset + i] = random.random()


        # Static/Dynamic // Rigid/Elastic
        offset = offset + HyperParameters.MaximumAttributeCount
        aggregate[offset] = 1.0
        aggregate[offset + 1] = 0.0

        # Effects
        offset = offset + 2
        aggregate[offset] = random.random()
        aggregate[offset + 1] = random.random()
        aggregate[offset + 2] = 0.5

        # External
        aggregate[offset + 3] = random.random()
        aggregate[offset + 4] = random.random()
        aggregate[offset + 5] = 0.5


        totalAccel = (np.array([aggregate[offset], aggregate[offset + 1]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale
        totalAccel = totalAccel + (np.array([aggregate[offset + 3], aggregate[offset + 4]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale

        ######### RECEIVER 
        offsetV = HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount
        for i in range(HyperParameters.MaximumAttributeCount):
            # Attributes:
            attributes[i] = aggregate[i] + ((aggregate[offsetV + i] * 2.0) - 1.0) * HyperParameters.TimeStep 
            # For non-Positions
            # Accelerations:
            attributes[HyperParameters.MaximumAttributeCount + i] = 0.5

        # Only apply force acceleration to position
        attributes[self._xPosOffset] = attributes[self._xPosOffset] + 0.5 * totalAccel[0] * HyperParameters.TimeStep * HyperParameters.TimeStep
        attributes[self._yPosOffset] = attributes[self._yPosOffset] + 0.5 * totalAccel[1] * HyperParameters.TimeStep * HyperParameters.TimeStep

        attributes[HyperParameters.MaximumAttributeCount + self._xPosOffset] = ((totalAccel[0] / HyperParameters.AccelerationScale) + 1.0) * 0.5
        attributes[HyperParameters.MaximumAttributeCount + self._yPosOffset] = ((totalAccel[1] / HyperParameters.AccelerationScale) + 1.0) * 0.5

        return aggregate, attributes


    def generateTestRelation(self, aX, aY, aVX, aVY, aMass, bX, bY, bVX, bVY, bMass):
        # Triplet Format:
        # Sender   -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic
        # Receiver -- Symbol | Attributes | Velocities | Static/Dynamic | Rigid/Elastic 
        # Relation -- Distance | Degrees-Of-Freedom | Sender Normal | Receiver Normal

        # Effect Format:
        # Acceleration Vector | Angle Acceleration Vector

        triplet = [0.0] * RelationTriplet.tripletLength()
        effect  = [0.0] * HyperParameters.DegreesOfFreedom

        ######### TRIPLET
        totalObjectEntries = (HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2)
        
        positionA = np.array([aX, aY])
        positionB = np.array([bX, bY])
        distanceVector = np.subtract(positionB, positionA)
        triplet[2 * totalObjectEntries] = (np.linalg.norm(distanceVector) - (aMass + bMass) * 0.5)
        differenceVector = distanceVector / (np.linalg.norm(distanceVector))

        # Filling Sender Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[HyperParameters.MaximumSymbolCount + i] = random.random()
        triplet[HyperParameters.MaximumSymbolCount + self._xPosOffset] = positionA[0]
        triplet[HyperParameters.MaximumSymbolCount + self._yPosOffset] = positionA[1]
        triplet[HyperParameters.MaximumSymbolCount + self._sizeOffset] = aMass
        triplet[HyperParameters.MaximumSymbolCount + self._arOffset] = 1.0
        
        # Filling Receiver Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + i] = random.random()
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._xPosOffset] = positionB[0]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._yPosOffset] = positionB[1]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._sizeOffset] = bMass
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._arOffset] = 1.0
        
        # Filling Sender Velocity
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + i] = 0.5
        triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._xPosOffset] = (aVX + 1.0) / 2.0
        triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._yPosOffset] = (aVY + 1.0) / 2.0
        
        # Filling Receiver Velocity
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[totalObjectEntries + HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + i] = 0.5
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._xPosOffset] = (bVX + 1.0) / 2.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._yPosOffset] = (bVY + 1.0) / 2.0

        # TODO:  The following are fixed for testing only Circles with Dynamic, Rigid Collisions:
        # Circle
        triplet[0] = 1.0
        triplet[totalObjectEntries] = 1.0
        # Static / Dynamic
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
        # Rigid / Elastic
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 0.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 0.0
        # Degrees-Of-Freedom
        triplet[2 * totalObjectEntries + 1] = 1.0
        triplet[2 * totalObjectEntries + 2] = 1.0
        triplet[2 * totalObjectEntries + 3] = 1.0
        # Sender Normal
        triplet[2 * totalObjectEntries + 4] = (differenceVector[0] + 1.0) / 2.0
        triplet[2 * totalObjectEntries + 5] = (differenceVector[1] + 1.0) / 2.0
        # Receiver Normal
        triplet[2 * totalObjectEntries + 6] = (-differenceVector[0] + 1.0) / 2.0
        triplet[2 * totalObjectEntries + 7] = (-differenceVector[1] + 1.0) / 2.0


        ######### EFFECT

        # We take the effects to be the sum of a Force F over time delta-t, i.e. Impuls I = F * delta-t
        velocityA = np.array([aVX, aVY])
        velocityB = np.array([bVX, bVY])
        hasCollision = False

        if triplet[2 * totalObjectEntries] < 0.01:
            hasCollision = True

        if np.dot(velocityA, differenceVector) < 0 and (np.dot(velocityB, differenceVector) > 0 or np.linalg.norm(velocityB) < np.linalg.norm(velocityA)):
            hasCollision = False
        elif np.dot(velocityB, -differenceVector) < 0 and (np.dot(velocityA, -differenceVector) > 0 or np.linalg.norm(velocityA) < np.linalg.norm(velocityB)):
            hasCollision = False

        if hasCollision is True:
            tempB = np.dot((velocityB - velocityA), distanceVector) / (math.pow(np.linalg.norm(distanceVector), 2.0))
            resultVelocityB = velocityB - (2 * aMass / (aMass + bMass)) * tempB * distanceVector
            resultAccelB = (resultVelocityB - velocityB) / HyperParameters.TimeStep

            # Scaling the Force Vectors
            effect[0] = (resultAccelB[0] / HyperParameters.AccelerationScale + 1.0) / 2.0
            effect[1] = (resultAccelB[1] / HyperParameters.AccelerationScale + 1.0) / 2.0
            effect[2] = 0.5
        else:
            effect[0] = 0.5
            effect[1] = 0.5
            effect[2] = 0.5

        return triplet, effect

