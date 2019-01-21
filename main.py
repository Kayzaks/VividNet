from CapsuleMemory import CapsuleMemory
from Attribute import Attribute
from AttributeType import AttributeType
from AttributeType import AttributeLexical
from AttributePool import AttributePool
from GraphicsUserInterface import GraphicsUserInterface
from Utility import Utility
from NeuralNet import NeuralNet

import numpy as np
import random
import math


testUI = GraphicsUserInterface()
attrPool = AttributePool()

attrPool.createType("X-Pos", AttributeLexical.Preposition)
attrPool.createType("Y-Pos", AttributeLexical.Preposition)
attrPool.createType("Radius", AttributeLexical.Preposition)

attrPool.createType("Theta", AttributeLexical.Adjective)
attrPool.createType("Squish", AttributeLexical.Adjective)

attrPool.createType("Pixel", AttributeLexical.Pixel)

attrIn = []  # Pixels
attrOut = [] # Semantics

pixelsAcross = 28

for xx in range(pixelsAcross):
    for yy in range(pixelsAcross):
        attrIn.append(attrPool.createAttribute("Pixel"))

attrOut.append(attrPool.createAttribute("X-Pos"))
attrOut.append(attrPool.createAttribute("Y-Pos"))
attrOut.append(attrPool.createAttribute("Radius"))
attrOut.append(attrPool.createAttribute("Theta"))
attrOut.append(attrPool.createAttribute("Squish"))

testMem = CapsuleMemory()

inMap = {}
outMap = {}
inMap2 = {}
outMap2 = {}

for idx, attr in enumerate(attrIn):
    inMap[idx] = attr
    inMap2[attr] = idx

for idx, attr in enumerate(attrOut):
    outMap[idx] = attr
    outMap2[attr] = idx

testMem.inferXAttributes(attrIn)
testMem.inferYAttributes(attrOut)



def windowFunction(x):
    try:
        if abs(x) < 1.0:
            return 1.0
        elif abs(x) < 1.5:
            return (1.0 - (abs(x) - 1.0) * 2.0)
        else:
            return 0.0
    except OverflowError:
        return 0.0

# Normalizing the Attributes using these values
circleOffset = 4
circleRange = pixelsAcross - (2*circleOffset)
circleRadius = 12

gFunction = (lambda attr: 
                        [windowFunction(math.sqrt(
                            math.pow(((x - (attr[0] * circleRange + circleOffset))*math.cos(-attr[3] * 3.14)-((pixelsAcross-y) - (attr[1] * circleRange + circleOffset))*math.sin(-attr[3] * 3.14))/attr[4],2)+
                            math.pow( (x - (attr[0] * circleRange + circleOffset))*math.sin(-attr[3] * 3.14)+((pixelsAcross-y) - (attr[1] * circleRange + circleOffset))*math.cos(-attr[3] * 3.14),2))
                            -attr[2] * circleRadius)
                            for y, x in np.ndindex((pixelsAcross, pixelsAcross))])

testMem.setLambdaKnownG(lambda : [random.random(), random.random(), (random.random() * 0.9) + 0.1, random.random(), (random.random() * 0.9) + 0.1], 
                        gFunction,
                        outMap, inMap)



testNN = NeuralNet(inMap2, outMap2, "TestNN")

# testNN.trainFromData(testMem, True)



batchX, batchY = testMem.nextBatch(1, inMap2, outMap2)
inputs = Utility.mapDataOneWay(batchX[0], inMap)
outputs = testNN.forwardPass(inputs)

pixels = []
for key, val in inputs.items():
    pixels.append(val)


detectedAttributes = [0.0] * len(outputs) 
for key, val in outputs.items():
    print(key.getName() + " : " + str(val) + " (Expected: " + str(Utility.mapDataOneWayRev(batchY[0], outMap2)[key])+ ")")
    detectedAttributes[outMap2[key]] = val

pixels2 = gFunction(detectedAttributes)

testUI.drawArrayCompare("Actual", "Detected", pixels, pixels2, pixelsAcross, pixelsAcross)