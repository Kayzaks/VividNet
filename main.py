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
from scipy import misc
from scipy.fftpack import fft, dct

currentShape = Shapes.Box

def doCompare(filename : str):
    image = misc.imread(filename, "L")
    image = np.asarray(image).astype(np.float32)
    
    pixels = np.empty(width * height * 3)

    for yy in range(height):
        for xx in range(width):
            pixels[(yy * width + xx) * 3] = image[yy][xx] * (1.0/255.0)
            pixels[(yy * width + xx) * 3 + 1] = float(xx) / float(width)
            pixels[(yy * width + xx) * 3 + 2] = float(yy) / float(height)

    inputs = Utility.mapDataOneWay(pixels, outMapIdxAttr)
    outputsBox = testNN.forwardPass(inputs)
    outputsSphere = otherNN.forwardPass(inputs)

    pixels2 = renderer.renderShape(currentShape, Utility.mapDataOneWayDict(outputsBox, inMapIdxAttr), width, height, pixels)
    pixels3 = renderer.renderShape(Shapes.Sphere, Utility.mapDataOneWayDict(outputsSphere, inMapIdxAttr), width, height, pixels)

    dctArr1 = np.empty((width, height))
    dctArr2 = np.empty((width, height))
    dctArr3 = np.empty((width, height))
    
    for yy in range(height):
        for xx in range(width):
            pixels[(yy * width + xx) * 3 + 1] = pixels[(yy * width + xx) * 3]
            pixels[(yy * width + xx) * 3 + 2] = pixels[(yy * width + xx) * 3]
            pixels2[(yy * width + xx) * 3 + 1] = pixels2[(yy * width + xx) * 3]
            pixels2[(yy * width + xx) * 3 + 2] = pixels2[(yy * width + xx) * 3]
            pixels3[(yy * width + xx) * 3 + 1] = pixels3[(yy * width + xx) * 3]
            pixels3[(yy * width + xx) * 3 + 2] = pixels3[(yy * width + xx) * 3]
            dctArr1[xx][yy] = pixels[(yy * width + xx) * 3]
            dctArr2[xx][yy] = pixels2[(yy * width + xx) * 3]
            dctArr3[xx][yy] = pixels3[(yy * width + xx) * 3]

    t1 = dct(dct(dctArr1, axis=0), axis=1)
    t2 = dct(dct(dctArr2, axis=0), axis=1)
    t3 = dct(dct(dctArr3, axis=0), axis=1)
    t1 = np.divide(t1, np.mean(t1))
    t2 = np.divide(t2, np.mean(t2))
    t3 = np.divide(t3, np.mean(t3))

    print("---")
    print(str(np.square(np.subtract(t1, t2)).sum()) + " --- " + str(np.square(np.subtract(t1, t3)).sum()))
    print(str(np.dot(t1.flatten(), t2.flatten()) / np.dot(t1.flatten(), t1.flatten())) + " --- " + str(np.dot(t1.flatten(), t3.flatten()) / np.dot(t1.flatten(), t1.flatten())))
    print(str(np.dot(pixels, pixels2) / np.dot(pixels, pixels)) + " --- " + str(np.dot(pixels, pixels3) / np.dot(pixels, pixels)))
    print(str(np.linalg.norm(np.subtract(pixels, pixels2))) + " --- " + str(np.linalg.norm(np.subtract(pixels, pixels3))))

    testUI.drawArrayCompare3("Real", "Detected Box", "Detected Sphere", pixels, pixels2, pixels3, width, height)



if __name__ == '__main__':

    # TODO: "Make Shape Capsule"

    testUI = GraphicsUserInterface()
    attrPool = AttributePool()
    sphereCapsule = Capsule("Sphere-Shape")

    renderer = ShapesRenderer()
    renderer.createAttributesForShape(currentShape, sphereCapsule, attrPool)

    width = 28
    height = 28

    pixelCapsule = Capsule("PixelLayer-28-28")
    renderer.createAttributesForPixelLayer(width, height, pixelCapsule, attrPool)

    sphereCapsule.addNewRoute([pixelCapsule])


    outMapIdxAttr, outMapAttrIdx = renderer.getLambdaGOutputMap(currentShape, pixelCapsule, width, height)
    inMapIdxAttr, inMapAttrIdx = renderer.getLambdaGInputMap(currentShape, sphereCapsule)

    sphereCapsule._routes[0]._memory.setLambdaKnownG(
        (lambda : renderer.renderInputGenerator(currentShape, width, height)), 
        (lambda attr: renderer.renderShape(currentShape, attr, width, height)),
                            outMapIdxAttr, inMapIdxAttr)






    testNN = NeuralNet(outMapAttrIdx, inMapAttrIdx, "BoxModel2", False)
    testNN.setModelSplit(renderer.getModelSplit(currentShape))
    #testNN.trainFromData(sphereCapsule._routes[0]._memory, False, [0])

    otherNN = NeuralNet(outMapAttrIdx, inMapAttrIdx, "SphereModel2", False)
    otherNN.setModelSplit(renderer.getModelSplit(Shapes.Sphere))


    testUI = GraphicsUserInterface()

    for i in range(3):
        batchX, batchY = sphereCapsule._routes[0]._memory.nextBatch(1, outMapAttrIdx, inMapAttrIdx)
        inputs = Utility.mapDataOneWay(batchX[0], outMapIdxAttr)

        outputsBox = testNN.forwardPass(inputs)
        outputsSphere = otherNN.forwardPass(inputs)

        pixels1 = renderer.renderShape(currentShape, batchY[0], width, height)
        pixels2 = renderer.renderShape(currentShape, Utility.mapDataOneWayDict(outputsBox, inMapIdxAttr), width, height, pixels1)
        pixels3 = renderer.renderShape(Shapes.Sphere, Utility.mapDataOneWayDict(outputsSphere, inMapIdxAttr), width, height, pixels1)

        for yy in range(height):
            for xx in range(width):
                pixels1[(yy * width + xx) * 3 + 1] = pixels1[(yy * width + xx) * 3]
                pixels1[(yy * width + xx) * 3 + 2] = pixels1[(yy * width + xx) * 3]
                pixels2[(yy * width + xx) * 3 + 1] = pixels2[(yy * width + xx) * 3]
                pixels2[(yy * width + xx) * 3 + 2] = pixels2[(yy * width + xx) * 3]
                pixels3[(yy * width + xx) * 3 + 1] = pixels3[(yy * width + xx) * 3]
                pixels3[(yy * width + xx) * 3 + 2] = pixels3[(yy * width + xx) * 3]


        #testUI.drawArray(pixels1, width, height)
        testUI.drawArrayCompare3("Real", "Detected Box", "Detected Sphere", pixels1, pixels2, pixels3, width, height)

    
    for i in range(1, 11):
        doCompare("Tests/CLEVR_" + str(i) + ".png")
     

    for i in range(1, 19):
        doCompare("Tests/REAL_" + str(i) + ".png")


    





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