from CapsuleMemory import CapsuleMemory
from Attribute import Attribute
from AttributeType import AttributeType
from AttributeType import AttributeLexical
from AttributePool import AttributePool
from Capsule import Capsule
from ShapesRenderer import Shapes
from ShapesRenderer import ShapesRenderer
from GraphicsUserInterface import GraphicsUserInterface
from Utility import Utility
from NeuralNet import NeuralNet

import numpy as np
import random
import math


# TODO: "Make Shape Capsule"

testUI = GraphicsUserInterface()
attrPool = AttributePool()
sphereCapsule = Capsule("Sphere-Shape")

renderer = ShapesRenderer()
renderer.createAttributesForShape(Shapes.Sphere, sphereCapsule, attrPool)

sphereCapsule.setAttributeValue("X-Size", 1.0)
sphereCapsule.setAttributeValue("Y-Size", 1.0)
sphereCapsule.setAttributeValue("Z-Size", 1.0)

sphereCapsule.setAttributeValue("X-Rot", 0.5)
sphereCapsule.setAttributeValue("Y-Rot", 0.5)
sphereCapsule.setAttributeValue("Z-Rot", 0.0)

sphereCapsule.setAttributeValue("R-Color", 1.0)
sphereCapsule.setAttributeValue("G-Color", 1.0)
sphereCapsule.setAttributeValue("B-Color", 1.0)

sphereCapsule.setAttributeValue("DCTR-0-0", 0.0)
sphereCapsule.setAttributeValue("DCTR-0-1", 0.0)
sphereCapsule.setAttributeValue("DCTR-0-2", 0.0)
sphereCapsule.setAttributeValue("DCTR-1-0", 0.0)
sphereCapsule.setAttributeValue("DCTR-1-1", 0.0)
sphereCapsule.setAttributeValue("DCTR-1-2", 0.0)
sphereCapsule.setAttributeValue("DCTR-2-0", 0.0)
sphereCapsule.setAttributeValue("DCTR-2-1", 0.0)
sphereCapsule.setAttributeValue("DCTR-2-2", 0.0)

sphereCapsule.setAttributeValue("Light-X-Dir", 0.3)
sphereCapsule.setAttributeValue("Light-Y-Dir", 0.7)
sphereCapsule.setAttributeValue("Light-Z-Dir", 0.6)

sphereCapsule.setAttributeValue("Light-R-Color", 1.0)
sphereCapsule.setAttributeValue("Light-G-Color", 1.0)
sphereCapsule.setAttributeValue("Light-B-Color", 1.0)

width = 28
height = 28

pixelCapsule = Capsule("PixelLayer-28-28")
renderer.createAttributesForPixelLayer(width, height, pixelCapsule, attrPool)

sphereCapsule.addNewRoute([pixelCapsule])


outMapIdxAttr, outMapAttrIdx = renderer.getLambdaGOutputMap(Shapes.Sphere, pixelCapsule, width, height)
inMapIdxAttr, inMapAttrIdx = renderer.getLambdaGInputMap(Shapes.Sphere, sphereCapsule)

sphereCapsule._routes[0]._memory.setLambdaKnownG(
    (lambda : renderer.renderInputGenerator(Shapes.Sphere, width, height)), 
    (lambda attr: renderer.renderShape(Shapes.Sphere, attr, width, height)),
                        outMapIdxAttr, inMapIdxAttr)


testNN = NeuralNet(outMapAttrIdx, inMapAttrIdx, "TestNN", False)

testNN.trainFromData(sphereCapsule._routes[0]._memory, True)



batchX, batchY = sphereCapsule._routes[0]._memory.nextBatch(1, outMapAttrIdx, inMapAttrIdx)

inputs = Utility.mapDataOneWay(batchX[0], outMapIdxAttr)
outputs = testNN.forwardPass(inputs)


pixels1 = renderer.renderShape(Shapes.Sphere, batchY[0], width, height)
pixels2 = renderer.renderShape(Shapes.Sphere, Utility.mapDataOneWayDict(outputs, inMapIdxAttr), width, height)


testUI = GraphicsUserInterface()

testUI.drawArrayCompare("Real", "Detected", pixels1, pixels2, width, height)





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