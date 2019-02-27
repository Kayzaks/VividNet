
from enum import Enum
from numba import cuda, float32, int32

import numpy as np
import math
import copy 

from AttributePool import AttributePool
from Attribute import Attribute
from AttributeType import AttributeLexical





# -------------------- Kernel Helpers

def applyBackground(xx : float, yy : float, color : list, dctCoefficients : list, dctDimension : int, depth : float, smoothDepth : bool = False):
    if smoothDepth is False and depth >= 1.0:
        return color

    currentCol = [0.0] * 3
    outputCol = [0.0] * 3

    fDCTDim = float(dctDimension)
    fXX = xx * fDCTDim
    fYY = yy * fDCTDim

    for uu in range(dctDimension):
        for vv in range(dctDimension):

            currentCol[0] = dctCoefficients[uu * dctDimension + vv] * math.cos((2.0 * fXX + 1.0) * float(uu) * math.pi / (fDCTDim * 2)) * math.cos((2.0 * fYY + 1.0) * float(vv) * math.pi / (fDCTDim * 2))
            currentCol[1] = dctCoefficients[uu * dctDimension + vv] * math.cos((2.0 * fXX + 1.0) * float(uu) * math.pi / (fDCTDim * 2)) * math.cos((2.0 * fYY + 1.0) * float(vv) * math.pi / (fDCTDim * 2))
            currentCol[2] = dctCoefficients[uu * dctDimension + vv] * math.cos((2.0 * fXX + 1.0) * float(uu) * math.pi / (fDCTDim * 2)) * math.cos((2.0 * fYY + 1.0) * float(vv) * math.pi / (fDCTDim * 2))

            if vv == 0:
                currentCol[0] = currentCol[0] * 0.707107 
                currentCol[1] = currentCol[1] * 0.707107 
                currentCol[2] = currentCol[2] * 0.707107 
            if uu == 0:
                currentCol[0] = currentCol[0] * 0.707107 
                currentCol[1] = currentCol[1] * 0.707107 
                currentCol[2] = currentCol[2] * 0.707107 

            outputCol[0] = outputCol[0] + currentCol[0]
            outputCol[1] = outputCol[1] + currentCol[1]
            outputCol[2] = outputCol[2] + currentCol[2]

    outputCol[0] = ((outputCol[0] * 2.0 / fDCTDim) + 128.0) / 256.0
    outputCol[1] = ((outputCol[1] * 2.0 / fDCTDim) + 128.0) / 256.0
    outputCol[2] = ((outputCol[2] * 2.0 / fDCTDim) + 128.0) / 256.0

    if smoothDepth is True:
        outputCol[0] = outputCol[0] * depth + color[0] * (1 - depth)
        outputCol[1] = outputCol[1] * depth + color[1] * (1 - depth)
        outputCol[2] = outputCol[2] * depth + color[2] * (1 - depth)

    return outputCol


def applyFilters(color : list):
    # TODO: Apply Filters
    return color


@cuda.jit('float32(float32, float32, float32[:,:], int32)', device=True)
def cudaGreyDCT(xx, yy, dctMatrix, dctDimension):
     
    currentCol = 0.0
    color = 0.0    
    fDctDim = float32(dctDimension)
    fXX = xx * fDctDim
    fYY = yy * fDctDim

    for uu in range(dctDimension):
        for vv in range(dctDimension):

            currentCol = dctMatrix[uu, vv] * math.cos((2.0 * fXX + 1.0) * float32(uu) * math.pi / (fDctDim * 2.0)) * math.cos((2.0 * fYY + 1.0) * float32(vv) * math.pi / (fDctDim * 2.0))
            
            if vv == 0:
                currentCol = currentCol * 0.707107 
            if uu == 0:
                currentCol = currentCol * 0.707107 

            color = color + currentCol

    return ((color * 2.0 / fDctDim) + 128.0) / 256.0


# -------------------- User Defined
# Example Render Kernel for CUDA
@cuda.jit
def cudaKernelExample(ioArray, width, height, attributes):
    offset = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Color
    ioArray[offset, 0] = attributes[0] 
    ioArray[offset, 1] = attributes[0]
    ioArray[offset, 2] = attributes[0]

    # Depth
    ioArray[offset, 3] = 0.0


# Example Render Kernel for Python
def pythonKernelExample(xx : float, yy : float, width : float, height : float, attributes : list):
    # Color and Depth
    color = [0.0] * 4
    color[0:3] = applyBackground(xx, yy, [attributes[0], attributes[0], attributes[0]], [0.0, 0.0, 0.0, 0.0], 2, color[3], True)
    color[0:3] = applyFilters(color[0:3])
    return color


# Inhert and extend
class Primitives(Enum):
    def __int__(self):
        return self.value
# --------------------




        
class PrimitivesRenderer:

    def __init__(self, attributePool : AttributePool):
        self._kernels = {}
        self._attributeLayouts = {}

        self.definePrimitives(attributePool)


    # -------------------- User Defined


    def definePrimitives(self, attributePool : AttributePool):
        ## Example Attribute
        # primAttributes : dict = {}     # Index - (Name, Lexical)

        ## Index must match the one used in the kernel
        # primAttributes[0] = ("Alpha", AttributeLexical.NonTransmit)

        # self.setPrimitiveAttributes(Primitives.Unknown, attributePool, primAttributes)
        # self.setKernels({Primitives.Unknown : pythonKernelExample})
        return


    def renderInputGenerator(self, primitive : Primitives, width : int, height : int):
        # Example Input Generator
        outList = np.random.rand(len(self._attributeLayouts[primitive]))
        return outList


    def getModelSplit(self, primitive : Primitives):
        # Example Model Split
        return [0, len(self._attributeLayouts[primitive])]


    def processAttributes(self, attributes : list):
        # Example Attribute pre-processing for renderer
        return self.processDCTCoefficients(attributes, 0, 0)


    # --------------------


    def processDCTCoefficients(self, attributes : list, startIndex : int, dctDimension : int):
        # TODO: Max Intensity according to dimension?
        for uu in range(dctDimension):
            for vv in range(dctDimension):
                intensity =  (1024.0 / ((max(uu, vv) + 1)))  
                attributes[startIndex + (uu * dctDimension) + vv] = (attributes[startIndex + (uu * dctDimension) + vv] - 0.5) * intensity 
        
        return attributes


    def setKernels(self, kernelFunctions : dict):
        # kernelFunctions  # Primitive - Kernel
        self._kernels = kernelFunctions


    def setPixelLayerAttributePool(self, attributePool : AttributePool, width : int, height : int):
        for xx in range(width):
            for yy in range(height):
                attributePool.createType("PixelC-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)
                attributePool.createType("PixelX-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)
                attributePool.createType("PixelY-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)
                attributePool.createType("PixelD-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)


    def createAttributesForPixelLayer(self, width : int, height : int, capsule, attributePool : AttributePool):
        self.setPixelLayerAttributePool(attributePool, width, height)
        
        for xx in range(width):
            for yy in range(height):
                capsule.createAttribute("PixelC-" + str(xx) + "-" + str(yy), attributePool)
                capsule.createAttribute("PixelX-" + str(xx) + "-" + str(yy), attributePool)
                capsule.createAttribute("PixelY-" + str(xx) + "-" + str(yy), attributePool)
                capsule.createAttribute("PixelD-" + str(xx) + "-" + str(yy), attributePool)
    

    def getLambdaGOutputMap(self, capsule, width : int, height : int):
        mapIdxAttr : dict = {}  # Index - Attribute
        mapAttrIdx : dict = {}  # Attribute - Index

        index = 0
        for yy in range(height):
            for xx in range(width):
                mapIdxAttr[index]     = capsule.getAttributeByName("PixelC-" + str(xx) + "-" + str(yy))
                mapIdxAttr[index + 1] = capsule.getAttributeByName("PixelX-" + str(xx) + "-" + str(yy))
                mapIdxAttr[index + 2] = capsule.getAttributeByName("PixelY-" + str(xx) + "-" + str(yy))
                mapIdxAttr[index + 3] = capsule.getAttributeByName("PixelD-" + str(xx) + "-" + str(yy))
                index = index + 4

        for key, value in mapIdxAttr.items():
            mapAttrIdx[value] = key

        return (mapIdxAttr, mapAttrIdx)


    def inferDimensionsFromPixelLayer(self, capsule):
        # Find the Dimension from a Pixel-Layer Capsule by looking at its attributes
        maxWidth = 0
        maxHeight = 0
        pixelTypes = {}
        for attribute in capsule.getAttributes():
            splits = attribute.getName().split("-")
            pixelTypes[splits[0]] = True
            digits = [int(symbol) for symbol in splits if symbol.isdigit() == True]
            if len(digits) == 2:
                maxWidth = max(digits[0], maxWidth)
                maxHeight = max(digits[1], maxHeight)
            
        return (maxWidth + 1, maxHeight + 1, len(pixelTypes))


    def setPrimitiveAttributes(self, primitive : Primitives, attributePool : AttributePool, primitiveAttributes : dict):
        # primitiveAttributes  # Index - (Name, Lexical)
        for key, value in primitiveAttributes.items():
            attributePool.createType(value[0], value[1])

        self._attributeLayouts[primitive] = copy.copy(primitiveAttributes)

        
    def createAttributesForPrimitive(self, primitive : Primitives, capsule, attributePool : AttributePool):
        for key, value in self._attributeLayouts[primitive].items():
            capsule.createAttribute(value[0], attributePool)


    def getLambdaGInputMap(self, primitive : Primitives, capsule):
        mapIdxAttr : dict = {}  # Index - Attribute
        mapAttrIdx : dict = {}  # Attribute - Index

        for key, value in self._attributeLayouts[primitive].items():
            mapIdxAttr[key] = capsule.getAttributeByName(value[0])

        # Invert the mapping
        for key, value in mapIdxAttr.items():
            mapAttrIdx[value] = key

        return (mapIdxAttr, mapAttrIdx)


    def renderPrimitive(self, primitive : Primitives, attributes : list, width : int, height : int, isTraining : bool = False, altBackground = None, showDebug : bool = False):
        # Input Attributes as Lambda G Mapping

        transferAttributes = np.asarray(self.processAttributes(copy.copy(attributes)))

        # Python alternative
        # return self.runPythonKernel(primitive, transferAttributes, width, height, altBackground, 1.0, isTraining)

        # Cuda alternative
        return self.runCudaKernel(primitive, transferAttributes, width, height, altBackground, 1.0, isTraining)


    def runPythonKernel(self, primitive : Primitives, attributes : list, width : int, height : int, altBackground = None, altBackgroundCutoff = 1.0, isTraining : bool = False):
        pixels = np.zeros(height * width * 4, dtype=float)

        fwidth = float(width)
        fheight = float(height)

        for xx in range(width):
            for yy in range(height):
                color = self._kernels[primitive](float(xx) / fwidth, float(yy) / fheight, fwidth, fheight, attributes)
                
                if ( color[3] >= altBackgroundCutoff) and altBackground is not None:
                    pixels[(yy * width + xx) * 4]       = altBackground[(yy * width + xx) * 4]
                    pixels[(yy * width + xx) * 4 + 1]   = altBackground[(yy * width + xx) * 4 + 1]
                    pixels[(yy * width + xx) * 4 + 2]   = altBackground[(yy * width + xx) * 4 + 2]
                    if isTraining is False:
                        pixels[(yy * width + xx) * 4 + 3]   = altBackgroundCutoff
                    else:                        
                        pixels[(yy * width + xx) * 4 + 3]   = 0.0
                else:
                    pixels[(yy * width + xx) * 4]     = color[0]
                    pixels[(yy * width + xx) * 4 + 1] = color[1]
                    pixels[(yy * width + xx) * 4 + 2] = color[2]
                    if isTraining is False:
                        pixels[(yy * width + xx) * 4 + 2] = color[2]
                    else:                        
                        pixels[(yy * width + xx) * 4 + 3]   = 0.0

        return pixels


    def runCudaKernel(self, primitive : Primitives, attributes : list, width : int, height : int, altBackground = None, altBackgroundCutoff = 1.0, isTraining : bool = False):
        pixels = np.zeros(height * width * 4, dtype=float)

        deviceAttributes    = cuda.to_device(attributes)
        returnImage         = np.zeros((width * height, 4), dtype=np.float32)
        threadsperblock     = 32 
        blockspergrid       = (width * height + (threadsperblock - 1)) // threadsperblock
 
        (self._kernels[primitive])[blockspergrid, threadsperblock](returnImage, width, height, deviceAttributes)
 
        for xx in range(width):
            for yy in range(height):
 
                if ( returnImage[xx * height + yy][3] >= altBackgroundCutoff) and altBackground is not None:
                    pixels[(yy * width + xx) * 4]       = altBackground[(yy * width + xx) * 4]
                    pixels[(yy * width + xx) * 4 + 1]   = altBackground[(yy * width + xx) * 4 + 1]
                    pixels[(yy * width + xx) * 4 + 2]   = altBackground[(yy * width + xx) * 4 + 2]
                    if isTraining is False:
                        pixels[(yy * width + xx) * 4 + 3]   = altBackgroundCutoff
                    else:                        
                        pixels[(yy * width + xx) * 4 + 3]   = 0.0
                else:
                    pixels[(yy * width + xx) * 4]     = returnImage[xx * height + yy][0]
                    pixels[(yy * width + xx) * 4 + 1] = returnImage[xx * height + yy][1]
                    pixels[(yy * width + xx) * 4 + 2] = returnImage[xx * height + yy][2]
                    if isTraining is False:
                        pixels[(yy * width + xx) * 4 + 3] = returnImage[xx * height + yy][3] 
                    else:                        
                        pixels[(yy * width + xx) * 4 + 3]   = 0.0

        return pixels