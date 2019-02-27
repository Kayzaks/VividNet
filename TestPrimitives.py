
# -------- These are some example primitive Capsules only meant for testing ---------

from numba import cuda, float32, int32
from Utility import Utility

import numpy as np
import random
import math

from PrimitivesRenderer import Primitives
from PrimitivesRenderer import PrimitivesRenderer
from PrimitivesRenderer import cudaGreyDCT
from PrimitivesRenderer import applyFilters
from AttributeType import AttributeLexical
from AttributePool import AttributePool

@cuda.jit('float32(float32, float32)', device=True)
def cudaWindowFunction(x, width):
    fullSupport = width
    linearSupport = width + 0.025
    if abs(x) < fullSupport:
        return 1.0
    elif abs(x) < linearSupport:
        return (1.0 - (abs(x) - fullSupport) / (linearSupport - fullSupport))
    else:
        return 0.0


@cuda.jit
def cudaKernelCircle(ioArray, width, height, attributes):
    offset = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    xx = float32(offset / height) / float32(width)
    yy = float32(offset % height) / float32(height) 

    FF = cuda.local.array(shape=(28, 28), dtype=float32)
    
    dctDim = 8
    index = 0
    for uu in range(dctDim):
        for vv in range(dctDim):
            FF[uu, vv] = attributes[7 + index]
            index = index + 1

    intensity1 = ((xx - (attributes[0])) * math.cos(-attributes[3] * math.pi)-(yy - (attributes[1])) * math.sin(-attributes[3] * math.pi)) / attributes[4]
    intensity2 = (xx - (attributes[0])) * math.sin(-attributes[3] * math.pi)+(yy - (attributes[1])) * math.cos(-attributes[3] * math.pi)
    intensity =  cudaWindowFunction(math.sqrt(intensity1 * intensity1 + intensity2 * intensity2) - attributes[2] * 0.5, (attributes[6] * 0.1) + 0.025)

    depth = 1.0 - intensity

    intensity =  intensity * attributes[5]
    background = cudaGreyDCT(xx, yy, FF, dctDim)

    # Color
    ioArray[offset, 0] = intensity * (1.0 - depth) + background * depth
    ioArray[offset, 1] = xx 
    ioArray[offset, 2] = yy 

    # Depth
    ioArray[offset, 3] = depth


@cuda.jit
def cudaKernelSquare(ioArray, width, height, attributes):
    offset = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    xx = float32(offset / height) / float32(width)
    yy = float32(offset % height) / float32(height) 

    FF = cuda.local.array(shape=(28, 28), dtype=float32)
    
    dctDim = 8
    index = 0
    for uu in range(dctDim):
        for vv in range(dctDim):
            FF[uu, vv] = attributes[7 + index]
            index = index + 1

    intensity1 = ((xx - (attributes[0])) * math.cos(-attributes[3] * math.pi)-(yy - (attributes[1])) * math.sin(-attributes[3] * math.pi)) / attributes[4]
    intensity2 = (xx - (attributes[0])) * math.sin(-attributes[3] * math.pi)+(yy - (attributes[1])) * math.cos(-attributes[3] * math.pi)
    intensity1 = abs(intensity1) - attributes[2] * 0.5 
    intensity2 = abs(intensity2) - attributes[2] * 0.5
    intensity =  cudaWindowFunction( max(intensity1, 0.0) + max(intensity2, 0.0) + min(max(intensity1, intensity2),0.0), (attributes[6] * 0.1) + 0.025)

    depth = 1.0 - intensity

    intensity =  intensity * attributes[5]
    background = cudaGreyDCT(xx, yy, FF, dctDim)

    # Color
    ioArray[offset, 0] = intensity * (1.0 - depth) + background * depth
    ioArray[offset, 1] = xx 
    ioArray[offset, 2] = yy 

    # Depth
    ioArray[offset, 3] = depth



DCTDIMENSION = 8
DCTOFFSET = 7


class TestPrimitives(Primitives):
    Circle = 0
    Square = 1

        
class TestRenderer(PrimitivesRenderer):
    def definePrimitives(self, attributePool : AttributePool):
        primAttributes : dict = {}     # Index - (Name, Lexical)

        # Circle
        primAttributes[0] = ("Position-X", AttributeLexical.Preposition)
        primAttributes[1] = ("Position-Y", AttributeLexical.Preposition)
        primAttributes[2] = ("Radius", AttributeLexical.Preposition)
        primAttributes[3] = ("Rotation", AttributeLexical.Preposition)
        primAttributes[4] = ("Aspect-Ratio", AttributeLexical.Preposition)
        
        primAttributes[5] = ("Intensity", AttributeLexical.Adjective)
        primAttributes[6] = ("Strength", AttributeLexical.Adjective)

        dctDim = DCTDIMENSION        
        index = DCTOFFSET
        for uu in range(dctDim):
            for vv in range(dctDim):
                primAttributes[index] = ("BackgroundDCT-" + str(uu) + "-" + str(vv), AttributeLexical.NonTransmit)
                index = index + 1
                
        self.setPrimitiveAttributes(TestPrimitives.Circle, attributePool, primAttributes)

        # Square
        primAttributes[2] = ("Size", AttributeLexical.Preposition)     
        self.setPrimitiveAttributes(TestPrimitives.Square, attributePool, primAttributes)


        self.setKernels({
                        TestPrimitives.Square : cudaKernelSquare,
                        TestPrimitives.Circle : cudaKernelCircle,
                        })


    def renderInputGenerator(self, primitive : Primitives, width : int, height : int):
        outList = np.random.rand(len(self._attributeLayouts[primitive]))

        # Center of Primitive away from edge
        # outList[0] = min(max(0.1, outList[0]), 0.9)
        # outList[1] = min(max(0.1, outList[1]), 0.9)
        # Center of Primitive fixated at center
        outList[0] = 0.5
        outList[1] = 0.5
        # Minimum Size
        outList[2] = max(0.1, outList[2])
        outList[4] = max(0.25, outList[4])
        # No "invisible" Primitive
        outList[6] = max(0.1, outList[6])


        return outList


    def processAttributes(self, attributes : list):
        return self.processDCTCoefficients(attributes, DCTOFFSET, DCTDIMENSION)

        
    def getModelSplit(self, primitive : Primitives):
        # Example Model Split
        return [0, DCTOFFSET, len(self._attributeLayouts[primitive])]
