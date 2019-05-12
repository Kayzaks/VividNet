

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
        
        collisionType = random.randint(0, 100)

        # Collision?
        if collisionType % 2 == 0:
            # Yes
            triplet[2 * totalObjectEntries] = random.random() * 0.05
        else:
            # No
            triplet[2 * totalObjectEntries] = random.random() * 0.6 + 0.05

        massSizeA = random.random() * 0.4 + 0.2
        massSizeB = random.random() * 0.4 + 0.2

        positionA = np.array([random.random(), random.random()])
        differenceVector = np.array([random.random() - 0.5, random.random() - 0.5])
        differenceVector = differenceVector / np.linalg.norm(differenceVector)
        distanceVector = differenceVector * (triplet[2 * totalObjectEntries] + massSizeA + massSizeB)
        positionB = np.add(positionA, distanceVector)

        # Slightly Off-Center Interactions
        offRotA = random.random() * 0.3 - 0.15
        offRotB = random.random() * 0.3 - 0.15

        velocityA = random.random() * 0.2 * np.array([differenceVector[0] * math.cos(offRotA) - differenceVector[1] * math.sin(offRotA), differenceVector[0] * math.cos(offRotA) + differenceVector[1] * math.sin(offRotA)])
        velocityB = random.random() * 0.2 * np.array([-differenceVector[0] * math.cos(offRotB) + differenceVector[1] * math.sin(offRotB), -differenceVector[0] * math.cos(offRotB) - differenceVector[1] * math.sin(offRotB)])

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
        triplet[HyperParameters.MaximumSymbolCount + self._xPosOffset] = (velocityA[0] + 1.0) / 2.0
        triplet[HyperParameters.MaximumSymbolCount + self._yPosOffset] = (velocityA[1] + 1.0) / 2.0
        
        # Filling Receiver Velocity
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._xPosOffset] = (velocityB[0] + 1.0) / 2.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._yPosOffset] = (velocityB[1] + 1.0) / 2.0

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
        triplet[2 * totalObjectEntries + 1] = 2.0 / float(HyperParameters.DegreesOfFreedom)
        # Sender Normal
        triplet[2 * totalObjectEntries + 2] = (differenceVector[0] + 1.0) / 2.0
        triplet[2 * totalObjectEntries + 3] = (differenceVector[1] + 1.0) / 2.0
        # Receiver Normal
        triplet[2 * totalObjectEntries + 4] = (-differenceVector[0] + 1.0) / 2.0
        triplet[2 * totalObjectEntries + 5] = (-differenceVector[1] + 1.0) / 2.0


        ######### EFFECT

        # We take the effects to be the sum of a Force F over time delta-t, i.e. Impuls I = F * delta-t
        
        tempB = np.cross((velocityB - velocityA), distanceVector) / np.linalg.norm(distanceVector)
        resultVelocityB = velocityB - (2 * massSizeA / (massSizeA + massSizeB)) * tempB * distanceVector
        resultAccelB = (resultVelocityB - velocityB) / HyperParameters.TimeStep

        # Scaling the Force Vectors
        effect[0] = (resultAccelB[0] * massSizeB / 10.0 + 1.0) / 2.0
        effect[1] = (resultAccelB[1] * massSizeB / 10.0 + 1.0) / 2.0

        return triplet, effect


    def generateInteraction(self):
        # Aggregate Format:
        # Receiver -- Attributes | Symbol | Velocities
        # Effects  -- Summed Effect Acceleration Vector | Summed Effect Angle Acceleration Vector
        # External -- External Acceleration Vector | External Angle Acceleration Vector

        # Attributes Format:
        # Receiver -- Attributes

        aggregate  = [0.0] * (HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + HyperParameters.DegreesOfFreedom * 2) 
        attributes = [0.0] * HyperParameters.MaximumAttributeCount 


        ######### AGGREGATE
        # Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            aggregate[i] = random.random()
        position = np.array([aggregate[self._xPosOffset], aggregate[self._yPosOffset]])
        mass = aggregate[self._sizeOffset]
        aggregate[self._arOffset] = 1.0

        # Symbol
        aggregate[HyperParameters.MaximumAttributeCount] = 1.0

        # Velocities
        offset = HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount
        aggregate[offset + self._xPosOffset] = random.random()
        aggregate[offset + self._yPosOffset] = random.random()
        velocity = 2.0 * np.array([aggregate[offset + self._xPosOffset], aggregate[offset + self._yPosOffset]]) - 1.0


        # Effects
        offset = offset + HyperParameters.MaximumAttributeCount 
        aggregate[offset] = random.random()
        aggregate[offset + 1] = random.random()

        # External
        aggregate[offset + 4] = random.random()
        aggregate[offset + 5] = random.random()


        totalForce = np.array([aggregate[offset] * 2.0 - 1.0, aggregate[offset + 1] * 2.0 - 1.0])
        totalForce = totalForce + np.array([aggregate[offset + 4] * 2.0 - 1.0, aggregate[offset + 5] * 2.0 - 1.0])

        ######### RECEIVER ATTRIBUTE
        for i in range(HyperParameters.MaximumAttributeCount):
            attributes[i] = aggregate[i]

        attributes[self._xPosOffset] = aggregate[self._xPosOffset] + velocity[0] * HyperParameters.TimeStep 
        attributes[self._yPosOffset] = aggregate[self._yPosOffset] + velocity[1] * HyperParameters.TimeStep 

        attributes[self._xPosOffset] = attributes[self._xPosOffset] + 0.5 * (totalForce[0] / mass) * HyperParameters.TimeStep * HyperParameters.TimeStep
        attributes[self._yPosOffset] = attributes[self._yPosOffset] + 0.5 * (totalForce[1] / mass) * HyperParameters.TimeStep * HyperParameters.TimeStep






        return aggregate, attributes