

from PrimitivesPhysics import PrimitivesPhysics
from CapsuleNetwork import CapsuleNetwork
from Capsule import Capsule
from Observation import Observation
from RelationTriplet import RelationTriplet
from HyperParameters import HyperParameters
from AttributePool import AttributePool

import numpy as np
import math
import random
import scipy.misc


class TestPhysics(PrimitivesPhysics):

    def init(self):
        self._xPosOffset = self._attributePool.getAttributeOrderByName("Position-X")
        self._yPosOffset = self._attributePool.getAttributeOrderByName("Position-Y")
        self._sizeOffset = self._attributePool.getAttributeOrderByName("Size")
        self._rotOffset = self._attributePool.getAttributeOrderByName("Rotation")
        self._arOffset = self._attributePool.getAttributeOrderByName("Aspect-Ratio")
        self._intOffset = self._attributePool.getAttributeOrderByName("Intensity")
        self._strOffset = self._attributePool.getAttributeOrderByName("Strength")



    def generateInteractionSequence(self, capsNet : CapsuleNetwork, width : int, height : int, folder : str, idname : str):
        # Generate Images in the folder with name id + "." + sequence_index + file_format

        # 0 = No Interaction
        # 1 = Newtonian Collision
        interType = random.randint(0, 1)        

        # 0 = Image Before
        # 1 = Image at Interaction
        # 2 = Image After

        positionA = [None, None, None]
        positionB = [None, None, None]

        positionA[1] = np.array([0.5, 0.5]) #np.array([random.random(), random.random()])

        # TODO: Assuming Circles for now       
        massA = random.random() * 0.2 + 0.1
        massB = random.random() * 0.2 + 0.1

        intA = random.random() 
        intB = random.random() 
        strA = min(random.random(), (0.333333 - massA) * 10.0 ) 
        strB = min(random.random(), (0.333333 - massB) * 10.0 ) 
        rotA = random.random() 
        rotB = random.random() 

        awayDir = np.array([random.random() - 0.5, random.random() - 0.5])
        awayDir = awayDir / np.linalg.norm(awayDir)

        velMod = 0.5

        if 1 == 1: #interType == 10:
            # No Interaction
            maxDist = random.random()
            awayVec = awayDir * ((massA + massB + (strA + strB) * 0.1 ) * 0.5 + maxDist + 0.02)
            positionB[1] = positionA[1] + awayVec
            # Velocities
            velA = np.array([random.random() - 0.5, random.random() - 0.5]) * min(maxDist, velMod * random.random())
            velB = np.array([random.random() - 0.5, random.random() - 0.5]) * min(maxDist, velMod * random.random())

            positionA[0] = positionA[1] - velA
            positionA[2] = positionA[1] + velA
            
            positionB[0] = positionB[1] - velB
            positionB[2] = positionB[1] + velB
            
        elif interType == 1:
            # Interaction
            awayVec = awayDir * ((massA + massB + (strA + strB) * 0.1 ) * 0.5)
            positionB[1] = positionA[1] + awayVec
            # Velocities
            velA = np.array([random.random() - 0.5, random.random() - 0.5]) * velMod * random.random()
            velB = np.array([random.random() - 0.5, random.random() - 0.5]) * velMod * random.random()

            if np.dot(velA, velB) < 0:
                # Flying away from each other -> Reverse one Velocity
                velA = -velA

            if (np.dot(velA, awayDir) < 0 and np.dot(velB, awayDir) < 0 and np.linalg.norm(velB) < np.linalg.norm(velA)) or \
               (np.dot(velA, awayDir) > 0 and np.dot(velB, awayDir) > 0 and np.linalg.norm(velA) < np.linalg.norm(velB)):
                # A Flying away from B and B flying towards A (or B Flying away from A and A flying towards B)
                # Only collide if B (A) is faster than A (B), thus we switch velocities
                velTemp = velA
                velA = velB
                velB = velTemp

            positionA[0] = positionA[1] - velA
            positionB[0] = positionB[1] - velB
            
            tempB = np.dot((velB - velA), awayVec) / (math.pow(np.linalg.norm(awayVec), 2.0))
            resultVelB = velB - (2 * massA / (massA + massB)) * tempB * awayVec
            
            tempA = np.dot((velA - velB), -awayVec) / (math.pow(np.linalg.norm(awayVec), 2.0))
            resultVelA = velA - (2 * massB / (massA + massB)) * tempA * (-awayVec)

            positionA[2] = positionA[1] + resultVelA            
            positionB[2] = positionB[1] + resultVelB

        attributesA = [None, None, None]
        attributesB = [None, None, None]

        for i in range(3):
            attributesA[i] = np.zeros(HyperParameters.MaximumAttributeCount)
            attributesB[i] = np.zeros(HyperParameters.MaximumAttributeCount)

            attributesA[i][self._xPosOffset] = positionA[i][0]
            attributesA[i][self._yPosOffset] = positionA[i][1]
            attributesA[i][self._sizeOffset] = massA
            attributesA[i][self._intOffset] = intA
            attributesA[i][self._strOffset] = strA
            attributesA[i][self._rotOffset] = rotA
            attributesA[i][self._arOffset] = 1.0

            attributesB[i][self._xPosOffset] = positionB[i][0]
            attributesB[i][self._yPosOffset] = positionB[i][1]
            attributesB[i][self._sizeOffset] = massB
            attributesB[i][self._intOffset] = intB
            attributesB[i][self._strOffset] = strB
            attributesB[i][self._rotOffset] = rotB
            attributesB[i][self._arOffset] = 1.0

        # Render Images and Save

        circCaps = capsNet.getCapsuleByName("TestPrimitives.Circle")

        for i in range(3):
            attrDictA = {}
            for j in range(len(attributesA[i])):
                attrDictA[circCaps.getAttributeByName(self._attributePool.getAttributeNameByOrder(j))] = attributesA[i][j]

            attrDictB = {}
            for j in range(len(attributesB[i])):
                attrDictB[circCaps.getAttributeByName(self._attributePool.getAttributeNameByOrder(j))] = attributesB[i][j]

            observationA = Observation(circCaps, circCaps._routes[0], [], attrDictA, 1.0)
            observationB = Observation(circCaps, circCaps._routes[0], [], attrDictB, 1.0)
            obs = {circCaps : [observationA, observationB]}

            imageReal, ignore1, ignore2 = capsNet.generateImage(width, height, obs)

            pixels = [0.0] * (width * height * 3)
            
            for yy in range(height):
                for xx in range(width):
                    pixels[(yy * width + xx) * 3] = imageReal[(yy * width + xx) * 4]
                    pixels[(yy * width + xx) * 3 + 1] = imageReal[(yy * width + xx) * 4]
                    pixels[(yy * width + xx) * 3 + 2] = imageReal[(yy * width + xx) * 4]

            scipy.misc.imsave(folder + idname + "." + str(i) + ".png", np.reshape(pixels, [height, width, 3]))

        return


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
        totalObjectEntries = (HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2 * HyperParameters.DegreesOfFreedom)
        
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
        vMagA = random.random()
        vMagB = random.random()
            
        # Filling Sender Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[HyperParameters.MaximumSymbolCount + i] = random.random()
        triplet[HyperParameters.MaximumSymbolCount + self._xPosOffset] = positionA[0]
        triplet[HyperParameters.MaximumSymbolCount + self._yPosOffset] = positionA[1]
        triplet[HyperParameters.MaximumSymbolCount + self._sizeOffset] = massSizeA
        
        # Filling Receiver Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + i] = random.random()
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._xPosOffset] = positionB[0]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._yPosOffset] = positionB[1]
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + self._sizeOffset] = massSizeB
        
        # Filling Sender Velocity
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + i] = random.random()
        triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._xPosOffset] = (velocityA[0] + 1.0) / 2.0
        triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._yPosOffset] = (velocityA[1] + 1.0) / 2.0
        triplet[HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._rotOffset] = vMagA
        
        # Filling Receiver Velocity
        for i in range(HyperParameters.MaximumAttributeCount):
            triplet[totalObjectEntries + HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + i] = random.random()
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._xPosOffset] = (velocityB[0] + 1.0) / 2.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._yPosOffset] = (velocityB[1] + 1.0) / 2.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + HyperParameters.MaximumAttributeCount + self._rotOffset] = vMagB

        # Train for all Symbols
        senderSymbol = random.randint(0, HyperParameters.MaximumSymbolCount)
        receiverSymbol = random.randint(0, HyperParameters.MaximumSymbolCount)
        triplet[senderSymbol] = 1.0
        triplet[totalObjectEntries + receiverSymbol] = 1.0
        # Static / Dynamic
        # For testing purposes, we also train the case, where the receiver can only rotate
        windmill = random.randint(0, 100)

        if windmill >= 75:
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 1.0
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2] = 1.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 0.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 0.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2] = 1.0
            triplet[totalObjectEntries + receiverSymbol] = 0.0
            # Figure 8
            triplet[totalObjectEntries + 3] = 1.0
        elif windmill >= 50:
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 0.0
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 0.0
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2] = 1.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 1.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2] = 1.0     
            triplet[senderSymbol] = 0.0
            # Figure 8
            triplet[3] = 1.0      
        else:
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 1.0
            triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2] = 1.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount] = 1.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 1] = 1.0
            triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 2] = 1.0
        # Rigid / Elastic
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 3] = 0.0
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 4] = 0.0
        triplet[HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 5] = 0.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 3] = 0.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 4] = 0.0
        triplet[totalObjectEntries + HyperParameters.MaximumSymbolCount + 2 * HyperParameters.MaximumAttributeCount + 5] = 0.0
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
        
        if windmill >= 75:
            # Ball - Windmill interaction.
            # This is not real physics, just something that simulates plausible looking interactions. If we knew the real physics anyways,
            # we could just implement that.

            # Fake linear momentums from angular momentums.
            vMagB = ((vMagB * 2.0) - 1.0) / (massSizeB * 0.5)
            velocityB = vMagB * np.array([-differenceVector[1], differenceVector[0]])
            massSizeA = massSizeA * massSizeB / 2.0
            massSizeB = massSizeB * massSizeB / 6.0
        elif windmill >= 50:
            vMagA = ((vMagA * 2.0) - 1.0) / (massSizeA * 0.5)
            velocityA = vMagA * np.array([-differenceVector[1], differenceVector[0]])
            massSizeB = massSizeA * massSizeB / 2.0
            massSizeA = massSizeA * massSizeA / 6.0


        if not (np.dot(velocityA, differenceVector) < 0 and (np.dot(velocityB, differenceVector) > 0 or np.linalg.norm(velocityB) < np.linalg.norm(velocityA))) and \
            not (np.dot(velocityB, -differenceVector) < 0 and (np.dot(velocityA, -differenceVector) > 0 or np.linalg.norm(velocityA) < np.linalg.norm(velocityB))) and \
            hasCollision == True:

            tempB = np.dot((velocityB - velocityA), distanceVector) / (math.pow(np.linalg.norm(distanceVector), 2.0))
            resultVelocityB = velocityB - (2 * massSizeA / (massSizeA + massSizeB)) * tempB * distanceVector
            resultAccelB = (resultVelocityB - velocityB) / HyperParameters.TimeStep

            if windmill >= 75:
                angDir = np.array([-differenceVector[1], differenceVector[0]])
                effect[0] = 0.5
                effect[1] = 0.5
                effect[2] = ((np.linalg.norm(resultAccelB * angDir) / HyperParameters.AccelerationScale) + 1.0) / 2.0
            else:
                # Scaling the Force Vectors
                effect[0] = ((resultAccelB[0] / HyperParameters.AccelerationScale) + 1.0) / 2.0
                effect[1] = ((resultAccelB[1] / HyperParameters.AccelerationScale) + 1.0) / 2.0
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

        aggregate  = [0.0] * (HyperParameters.MaximumAttributeCount * 2 + HyperParameters.MaximumSymbolCount + HyperParameters.DegreesOfFreedom * 4) 
        attributes = [0.0] * HyperParameters.MaximumAttributeCount * 2


        ######### AGGREGATE
        # Attributes
        for i in range(HyperParameters.MaximumAttributeCount):
            aggregate[i] = random.random()

        # Symbol
        aggregate[HyperParameters.MaximumAttributeCount + random.randint(0, HyperParameters.MaximumSymbolCount)] = 1.0

        # Velocities
        offset = HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount
        for i in range(HyperParameters.MaximumAttributeCount):
            aggregate[offset + i] = random.random()


        # Static/Dynamic // Rigid/Elastic
        offset = offset + HyperParameters.MaximumAttributeCount

        aggregate[offset] = float(random.randint(0, 1))
        aggregate[offset + 1] = float(random.randint(0, 1))
        aggregate[offset + 2] = float(random.randint(0, 1))
        aggregate[offset + 3] = 0.0
        aggregate[offset + 4] = 0.0
        aggregate[offset + 5] = 0.0

        # Effects
        offset = offset + 2 * HyperParameters.DegreesOfFreedom
        aggregate[offset] = random.random()
        aggregate[offset + 1] = random.random()
        aggregate[offset + 2] = random.random()

        # External
        aggregate[offset + 3] = random.random()
        aggregate[offset + 4] = random.random()
        aggregate[offset + 5] = random.random()


        totalAccel = (np.array([aggregate[offset], aggregate[offset + 1], aggregate[offset + 2]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale
        totalAccel = totalAccel + (np.array([aggregate[offset + 3], aggregate[offset + 4], aggregate[offset + 5]]) * 2.0 - 1.0) * HyperParameters.AccelerationScale

        ######### RECEIVER 
        offsetV = HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount
        for i in range(HyperParameters.MaximumAttributeCount):
            # Attributes:
            attributes[i] = aggregate[i] + ((aggregate[offsetV + i] * 2.0) - 1.0) * HyperParameters.TimeStep 
            # For non-Positions
            # Accelerations:
            attributes[HyperParameters.MaximumAttributeCount + i] = 0.5

        # Only apply force acceleration to position and rotation
        attributes[self._xPosOffset] = attributes[self._xPosOffset] + 0.5 * totalAccel[0] * HyperParameters.TimeStep * HyperParameters.TimeStep
        attributes[self._yPosOffset] = attributes[self._yPosOffset] + 0.5 * totalAccel[1] * HyperParameters.TimeStep * HyperParameters.TimeStep
        attributes[self._rotOffset] = attributes[self._rotOffset] + 0.5 * totalAccel[2] * HyperParameters.TimeStep * HyperParameters.TimeStep
        
        attributes[HyperParameters.MaximumAttributeCount + self._xPosOffset] = ((totalAccel[0] / HyperParameters.AccelerationScale) + 1.0) * 0.5
        attributes[HyperParameters.MaximumAttributeCount + self._yPosOffset] = ((totalAccel[1] / HyperParameters.AccelerationScale) + 1.0) * 0.5
        attributes[HyperParameters.MaximumAttributeCount + self._rotOffset] = ((totalAccel[2] / HyperParameters.AccelerationScale) + 1.0) * 0.5

        # Static.. Undo changes
        if aggregate[2 * HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount] < 0.1:
            attributes[self._xPosOffset] = aggregate[self._xPosOffset]
            attributes[HyperParameters.MaximumAttributeCount + self._xPosOffset] = 0.5
        if aggregate[2 * HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + 1] < 0.1:
            attributes[self._yPosOffset] = aggregate[self._yPosOffset]
            attributes[HyperParameters.MaximumAttributeCount + self._yPosOffset] = 0.5
        if aggregate[2 * HyperParameters.MaximumAttributeCount + HyperParameters.MaximumSymbolCount + 2] < 0.1:
            attributes[self._rotOffset] = aggregate[self._rotOffset]
            attributes[HyperParameters.MaximumAttributeCount + self._rotOffset] = 0.5



        return aggregate, attributes
