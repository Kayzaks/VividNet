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

if __name__ == '__main__':

    # TODO: "Make Shape Capsule"

    testUI = GraphicsUserInterface()
    attrPool = AttributePool()
    sphereCapsule = Capsule("Sphere-Shape")

    renderer = ShapesRenderer()
    renderer.createAttributesForShape(Shapes.Sphere, sphereCapsule, attrPool)

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



    testNN = NeuralNet(outMapAttrIdx, inMapAttrIdx, "SphereModel", False)
    testNN.setModelSplit(renderer.getModelSplit(Shapes.Sphere))
    testNN.trainFromData(sphereCapsule._routes[0]._memory, False, [4])



    testUI = GraphicsUserInterface()


    
    from scipy import misc

    for i in range(1, 10):
        image = misc.imread("Tests/CLEVR_" + str(i) + ".png")
        image = np.asarray(image).astype(np.float32).flatten() * (1.0/255.0)

        inputs = Utility.mapDataOneWay(image, outMapIdxAttr)
        outputs = testNN.forwardPass(inputs)
        
        pixels2 = renderer.renderShape(Shapes.Sphere, Utility.mapDataOneWayDict(outputs, inMapIdxAttr), width, height)

        print(np.dot(image, pixels2) / np.dot(image, image))

        testUI.drawArrayCompare("Real", "Detected", image, pixels2, width, height)
    

    for i in range(3):
        batchX, batchY = sphereCapsule._routes[0]._memory.nextBatch(1, outMapAttrIdx, inMapAttrIdx)
        inputs = Utility.mapDataOneWay(batchX[0], outMapIdxAttr)

        outputs = testNN.forwardPass(inputs)

        pixels1 = renderer.renderShape(Shapes.Sphere, batchY[0], width, height)
        pixels2 = renderer.renderShape(Shapes.Sphere, Utility.mapDataOneWayDict(outputs, inMapIdxAttr), width, height)

        print(np.dot(pixels1, pixels2) / np.dot(pixels1, pixels1))

        #testUI.drawArray(pixels1, width, height)
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