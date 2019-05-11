
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

@cuda.jit('float32(float32, float32, float32)', device=True)
def cudaWindowFunction(x, width, falloff):
    fullSupport = width
    linearSupport = width + falloff 
    if abs(x) < fullSupport:
        return 1.0
    elif abs(x) < linearSupport:
        return (1.0 - (abs(x) - fullSupport) / (linearSupport - fullSupport))
    else:
        return 0.0


@cuda.jit('float32(float32, float32, float64[:])', device=True)
def cudaSDFCircle(xx, yy, attributes):
    tx = ((xx - (attributes[0])) * math.cos(-attributes[3] * math.pi * 2.0)-(yy - (attributes[1])) * math.sin(-attributes[3] * math.pi * 2.0)) / attributes[4]
    ty = (xx - (attributes[0])) * math.sin(-attributes[3] * math.pi * 2.0)+(yy - (attributes[1])) * math.cos(-attributes[3] * math.pi * 2.0)
    return cudaWindowFunction(math.sqrt(tx * tx + ty * ty) - attributes[2] * 0.5, (attributes[6] * 0.1) + 0.025, 0.025)
    
@cuda.jit('float32(float32, float32, float64[:])', device=True)
def cudaSDFSquare(xx, yy, attributes):
    tx = ((xx - (attributes[0])) * math.cos(-attributes[3] * math.pi * 2.0)-(yy - (attributes[1])) * math.sin(-attributes[3] * math.pi * 2.0)) / attributes[4]
    ty = (xx - (attributes[0])) * math.sin(-attributes[3] * math.pi * 2.0)+(yy - (attributes[1])) * math.cos(-attributes[3] * math.pi * 2.0)
    tx = abs(tx) - attributes[2] * 0.5 
    ty = abs(ty) - attributes[2] * 0.5
    return cudaWindowFunction( max(tx, 0.0) + max(ty, 0.0) + min(max(tx, ty),0.0), (attributes[6] * 0.1) + 0.025, 0.025)
    
@cuda.jit('float32(float32, float32, float64[:])', device=True)
def cudaSDFTriangle(xx, yy, attributes):
    tx = ((xx - (attributes[0])) * math.cos(-attributes[3] * math.pi * 2.0 + math.pi)-(yy - (attributes[1])) * math.sin(-attributes[3] * math.pi * 2.0 + math.pi)) / attributes[4]
    ty = (xx - (attributes[0])) * math.sin(-attributes[3] * math.pi * 2.0 + math.pi)+(yy - (attributes[1])) * math.cos(-attributes[3] * math.pi * 2.0 + math.pi)
    k = 1.732050    
    px = abs(2.0 * tx / attributes[2]) - 1.0
    py = 2.0 * ty / attributes[2] + 1.0/k
    if px + k * py > 0.0:
        ptemp = (-k*px - py) / 2.0
        px = (px - k*py) / 2.0
        py = ptemp
    if px < -2.0:
        px = -2.0
    if px > 0.0:
        px = 0.0
    px -= px
    return cudaWindowFunction( (-math.sqrt(px * px + py * py) * math.copysign(1.0, py)) * attributes[2], (attributes[6] * 0.1), 0.025)


@cuda.jit
def cudaKernel(ioArray, width, height, attributes, primitive, isTraining):
    offset = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    xx = float32(offset / height) / float32(width)
    yy = float32(offset % height) / float32(height) 

    intensity = 0.0
    
    if primitive == 0:
        intensity = cudaSDFCircle(xx, yy, attributes)
    elif primitive == 1:
        intensity = cudaSDFSquare(xx, yy, attributes)
    elif primitive == 2:
        intensity = cudaSDFTriangle(xx, yy, attributes)

    depth = 1.0 - intensity
    intensity = (intensity * attributes[5] * 0.4) + 0.6

    background = 0.0

    if len(attributes) > 8 and isTraining is True:
        for i in range(3):
            newBack = 0.0
            if attributes[7 + i * 8] > -0.1 and attributes[7 + i * 8] < 0.1:
                newBack = cudaSDFCircle(xx, yy, attributes[(8 + i * 8):])
            elif attributes[7 + i * 8] > 0.9 and attributes[7 + i * 8] < 1.1:
                newBack = cudaSDFSquare(xx, yy, attributes[(8 + i * 8):])
            elif attributes[7 + i * 8] > 1.9 and attributes[7 + i * 8] < 2.1:
                newBack = cudaSDFTriangle(xx, yy, attributes[(8 + i * 8):])
            
            newBack = ((newBack * attributes[8 + 5 + i * 8] * 0.4) + 0.6) * newBack
            background = max(background, newBack)
            

    # Color
    ioArray[offset, 0] = intensity * (1.0 - depth) + background * depth
    ioArray[offset, 1] = math.sqrt((xx - 0.5) * (xx - 0.5) + (yy - 0.5) * (yy - 0.5))
    angle = math.atan2((xx - 0.5), (yy - 0.5)) # atan2 Reversed to use y-axis as reference 
    if angle < 0.0:
        angle = 2 * math.pi + angle
    ioArray[offset, 2] = angle

    # Depth
    ioArray[offset, 3] = depth



class TestPrimitives(Primitives):
    Circle = 0
    Square = 1
    Triangle = 2

    NoPrimitive = -1

        
class TestRenderer(PrimitivesRenderer):

    def definePrimitives(self, attributePool : AttributePool):
        primAttributes : dict = {}     # Index - (Name, Lexical)

        # Circle
        primAttributes[0] = ("Position-X", AttributeLexical.Preposition)
        primAttributes[1] = ("Position-Y", AttributeLexical.Preposition)
        primAttributes[2] = ("Size", AttributeLexical.Preposition)          # Radius
        primAttributes[3] = ("Rotation", AttributeLexical.Preposition)
        primAttributes[4] = ("Aspect-Ratio", AttributeLexical.Preposition)
        
        primAttributes[5] = ("Intensity", AttributeLexical.Adjective)
        primAttributes[6] = ("Strength", AttributeLexical.Adjective)
                
        self.setPrimitiveAttributes(TestPrimitives.Circle, attributePool, primAttributes)
        self.addPrimitiveDimensions(TestPrimitives.Circle, 28, 28)

        # Square   
        self.setPrimitiveAttributes(TestPrimitives.Square, attributePool, primAttributes)
        self.addPrimitiveDimensions(TestPrimitives.Square, 28, 28)

        # Triangle   
        self.setPrimitiveAttributes(TestPrimitives.Triangle, attributePool, primAttributes)
        self.addPrimitiveDimensions(TestPrimitives.Triangle, 28, 28)


        self.setKernel(cudaKernel)


    def renderInputGenerator(self, primitive : Primitives, width : int, height : int):
        outList = np.random.rand(len(self._attributeLayouts[primitive]))

        # Center of Primitive away from edge
        # outList[0] = min(max(0.1, outList[0]), 0.9)
        # outList[1] = min(max(0.1, outList[1]), 0.9)
        # Center of Primitive fixated at center
        outList[0] = 0.5
        outList[1] = 0.5

        # Minimum Size
        outList[2] = max(0.2, outList[2])
        outList[4] = max(0.5, outList[4])
        # No "invisible" Primitive
        outList[6] = max(0.1, outList[6])

        if primitive == TestPrimitives.Circle:
            # Limit Rotations to 0 - Pi (Rotationally symmetric)
            outList[3] = outList[3] * 0.5
        if primitive == TestPrimitives.Square:
            # Limit Rotations to 0 - Pi (Rotationally symmetric)
            outList[3] = outList[3] * 0.25
        if primitive == TestPrimitives.Triangle:
            # Limit Rotations to 0 - Pi (Rotationally symmetric)
            outList[3] = outList[3] * 0.3333
            outList[4] = max(0.5, outList[4])

        return outList

    
    def processAttributes(self, attributes : list):
        # Add some random Background Primitives (3 in total)
        prims = [x - 1 for x in np.random.randint(4, size=3)] 
        extras = []
        for primitive in prims:
            currList = np.random.rand(7)

            currList[0] = (currList[0] - 0.1) * 1.2
            currList[1] = (currList[1] - 0.1) * 1.2

            # Minimum Size
            currList[2] = max(0.2, currList[2])
            currList[4] = max(0.5, currList[4])
            # No "invisible" Primitive
            currList[6] = max(0.1, currList[6])

            if primitive == int(TestPrimitives.Circle):
                # Limit Rotations to 0 - Pi (Rotationally symmetric)
                currList[3] = currList[3] * 0.5
            if primitive == int(TestPrimitives.Square):
                # Limit Rotations to 0 - Pi (Rotationally symmetric)
                currList[3] = currList[3] * 0.25
            if primitive == int(TestPrimitives.Triangle):
                # Limit Rotations to 0 - Pi (Rotationally symmetric)
                currList[3] = currList[3] * 0.3333
                currList[4] = max(0.5, currList[4])
            
            extras.append(float(primitive))
            extras.extend(currList)
        return np.concatenate((attributes, extras), axis=None)

