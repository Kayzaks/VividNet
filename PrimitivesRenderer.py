
from enum import Enum
from numba import cuda

import numpy as np

from AttributePool import AttributePool
from Attribute import Attribute
from AttributeType import AttributeLexical


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
    return [attributes[0], attributes[0], attributes[0], 0.0]


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
    # --------------------


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

        self._attributeLayouts[primitive] = primitiveAttributes

        
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


    def renderPrimitive(self, primitive : Primitives, attributes : list, width : int, height : int, altBackground = None, showDebug : bool = False):
        # Input Attributes as Lambda G Mapping

        transferAttributes = np.empty(len(attributes), dtype=np.float32)

        for index in range(len(attributes)):
            transferAttributes[index] = attributes[index]

        return self.runPythonKernel(primitive, transferAttributes, width, height, altBackground, 2.0)

        # Cuda alternative
        # return self.runCudaKernel(primitive, transferAttributes, width, height, altBackground, 2.0)


    def runPythonKernel(self, primitive : Primitives, attributes : list, width : int, height : int, altBackground = None, altBackgroundCutoff = 2.0):
        pixels = np.zeros(height * width * 4, dtype=float)

        for xx in range(width):
            for yy in range(height):
                color = self._kernels[primitive](float(xx), float(yy), float(width), float(height), attributes)
                
                if ( color[3] >= altBackgroundCutoff) and altBackground is not None:
                    pixels[(yy * width + xx) * 4]       = altBackground[(yy * width + xx) * 4]
                    pixels[(yy * width + xx) * 4 + 1]   = altBackground[(yy * width + xx) * 4 + 1]
                    pixels[(yy * width + xx) * 4 + 2]   = altBackground[(yy * width + xx) * 4 + 2]
                else:
                    pixels[(yy * width + xx) * 4]     = color[0]
                    pixels[(yy * width + xx) * 4 + 1] = color[1]
                    pixels[(yy * width + xx) * 4 + 2] = color[2]

                pixels[(yy * width + xx) * 4 + 3] = color[3]

        return pixels


    def runCudaKernel(self, primitive : Primitives, attributes : list, width : int, height : int, altBackground = None, altBackgroundCutoff = 2.0):
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
                else:
                    pixels[(yy * width + xx) * 4]     = returnImage[xx * height + yy][0]
                    pixels[(yy * width + xx) * 4 + 1] = returnImage[xx * height + yy][1]
                    pixels[(yy * width + xx) * 4 + 2] = returnImage[xx * height + yy][2]
 
                pixels[(yy * width + xx) * 4 + 3] = returnImage[xx * height + yy][3]
 

        return pixels